"""
GhostSignal — ML Training Pipeline
====================================
Trains two independent TF-IDF + Logistic Regression classifiers:

  1. SENTIMENT CLASSIFIER
     - Input: Amazon + Google reviews
     - Labels: derived from star ratings (1-2 -> Negative, 4-5 -> Positive)
     - 3-star reviews are excluded as ambiguous
     - Labels are RATING-DERIVED, not human sentiment annotations.

  2. AUTHENTICITY CLASSIFIER
     - Input: Amazon reviews only (labeled subset)
     - Labels: reviewer_labeled_fake==1 -> FAKE, reviewer_labeled_honest==1 -> HONEST
     - Records where both fields are 0 or null are excluded entirely
     - Labels are REVIEWER-ASSOCIATED, not verified per-review fraud labels.
       A reviewer flagged as fake has that label propagated to all their reviews.
     - Train/test split is grouped by reviewer_id to prevent data leakage.

Random seed: 42 (all splits and estimators)
Artifacts saved under: models/
"""

import json
import logging
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.model.evaluate import evaluate_classifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
AMAZON_JSONL = _ROOT / "data" / "processed" / "amazon_reviews_normalized.jsonl"
GOOGLE_JSONL = _ROOT / "data" / "processed" / "google_reviews_normalized.jsonl"
MODELS_DIR = _ROOT / "models"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Library code should not configure the root logger.
# The NullHandler prevents "No handlers could be found" warnings when this
# module is imported without any logging configured by the host application.
logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _rating_to_sentiment(rating: int) -> str | None:
    """
    Convert a 1-5 star rating to a binary sentiment label.
    Returns None for 3-star (ambiguous) records.
    Labels are RATING-DERIVED sentiment, not human annotations.
    """
    if rating in (1, 2):
        return "Negative"
    if rating in (4, 5):
        return "Positive"
    return None  # 3 stars — excluded


def load_sentiment_data() -> tuple[list[str], list[str], list[str]]:
    """
    Stream both JSONL files and build the sentiment dataset.

    Returns
    -------
    texts  : list of review text strings
    labels : list of 'Positive' or 'Negative'
    groups : list of author IDs (for group-aware splitting only)
    """
    texts: list[str] = []
    labels: list[str] = []
    groups: list[str] = []

    total_read = 0
    skipped_no_content = 0
    skipped_ambiguous = 0

    for path in (AMAZON_JSONL, GOOGLE_JSONL):
        source_name = path.stem.split("_")[0]
        log.info("Loading sentiment data from %s ...", source_name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_read += 1
                content = record.get("content") or ""
                content = content.strip()
                if not content:
                    skipped_no_content += 1
                    continue

                rating = record.get("rating")
                if rating is None:
                    skipped_ambiguous += 1
                    continue

                label = _rating_to_sentiment(int(rating))
                if label is None:
                    skipped_ambiguous += 1
                    continue

                # Author is used only as a group key for splitting.
                author = record.get("author") or record.get("id") or f"_unknown_{total_read}"

                texts.append(content)
                labels.append(label)
                groups.append(str(author))

    log.info(
        "Sentiment data loaded: %d usable records "
        "(skipped: %d no-content, %d ambiguous/3-star)",
        len(texts),
        skipped_no_content,
        skipped_ambiguous,
    )
    return texts, labels, groups


def load_authenticity_data() -> tuple[list[str], list[str], list[str]]:
    """
    Stream the Amazon JSONL and build the authenticity dataset.

    Label assignment (reviewer-level, propagated to all their reviews):
      reviewer_labeled_fake == 1.0  ->  'FAKE'
      reviewer_labeled_honest == 1.0  ->  'HONEST'
      both 0.0, or either null  ->  EXCLUDED

    Labels describe the REVIEWER, not individual review content.
    reviewer_labeled_fake == 0.0 does NOT mean genuine; those records
    are discarded, not assigned to the HONEST class.

    Returns
    -------
    texts  : list of review text strings
    labels : list of 'FAKE' or 'HONEST'
    groups : list of reviewer_id strings (for group-aware splitting only)
    """
    texts: list[str] = []
    labels: list[str] = []
    groups: list[str] = []

    total_read = 0
    skipped_no_content = 0
    skipped_ambiguous = 0  # 0/0 and null cases

    log.info("Loading authenticity data from Amazon ...")
    with open(AMAZON_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_read += 1
            content = record.get("content") or ""
            content = content.strip()
            if not content:
                skipped_no_content += 1
                continue

            meta = record.get("metadata") or {}
            lf = meta.get("reviewer_labeled_fake")   # float | None
            lh = meta.get("reviewer_labeled_honest")  # float | None

            # Determine label — strict rules, no inference from negatives.
            if lf == 1.0 and lh == 0.0:
                label = "FAKE"
            elif lh == 1.0 and lf == 0.0:
                label = "HONEST"
            else:
                # Covers: both null, both 0, contradictory, partial null.
                skipped_ambiguous += 1
                continue

            # reviewer_id is used only as a group key for splitting.
            reviewer_id = (
                meta.get("reviewer_id")
                or record.get("author")
                or f"_unknown_{total_read}"
            )

            texts.append(content)
            labels.append(label)
            groups.append(str(reviewer_id))

    log.info(
        "Authenticity data loaded: %d usable records "
        "(skipped: %d no-content, %d ambiguous/excluded)",
        len(texts),
        skipped_no_content,
        skipped_ambiguous,
    )
    return texts, labels, groups


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _build_pipeline(class_weight: dict | str | None = "balanced") -> Pipeline:
    """
    Construct a TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    class_weight : passed directly to LogisticRegression. 'balanced'
                   re-weights inversely proportional to class frequency.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents="unicode",
                    analyzer="word",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                    C=1.0,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Group-aware split helper
# ---------------------------------------------------------------------------

def _group_split(
    texts: list[str],
    labels: list[str],
    groups: list[str],
    test_size: float = TEST_SIZE,
    return_test_groups: bool = False,
) -> tuple:
    """
    Split texts/labels into train/test sets guaranteeing that every group
    (reviewer_id or author) appears exclusively in either the training set or
    the test set, never both.

    A record-level fallback (train_test_split) is intentionally not provided.
    If GroupShuffleSplit cannot produce a valid partition, training is aborted
    with a RuntimeError.  Silently falling back to a record-level split would
    allow the same reviewer's reviews to appear in both train and test, which
    is data leakage for this task.

    Parameters
    ----------
    texts, labels, groups : aligned lists from a load_*_data() call
    test_size             : fraction of data to hold out (default TEST_SIZE)
    return_test_groups    : when True, appends the list of group IDs for the
                            test slice as a fifth return value, so the caller
                            can pass them to evaluate_classifier for
                            reviewer-level metrics without re-running the split.

    Returns
    -------
    X_train, X_test, y_train, y_test[, groups_test]

    Raises
    ------
    RuntimeError
        If GroupShuffleSplit fails to produce a group-disjoint split.  The
        error message includes the number of unique groups and a suggestion
        for how to resolve the problem.
    """
    X = np.array(texts)
    y = np.array(labels)
    g = np.array(groups)

    n_groups = len(set(g))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    try:
        train_idx, test_idx = next(gss.split(X, y, groups=g))
    except Exception as exc:
        raise RuntimeError(
            f"GroupShuffleSplit could not produce a group-disjoint train/test split "
            f"({n_groups} unique groups, {len(X)} records, test_size={test_size}). "
            f"Original error: {exc}. "
            f"Ensure the dataset has enough distinct groups to support the requested "
            f"split ratio. A record-level fallback is not used because it would "
            f"allow the same reviewer/author to appear in both train and test sets."
        ) from exc

    # Verify the split is truly group-disjoint before proceeding.
    train_groups = set(g[train_idx])
    test_groups  = set(g[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            f"GroupShuffleSplit produced a split with {len(overlap)} overlapping "
            f"group(s). This should not happen with a correct GroupShuffleSplit "
            f"call; it may indicate a bug in the sklearn version in use. "
            f"Aborting to prevent data leakage. Overlapping groups (first 5): "
            f"{list(overlap)[:5]}"
        )

    log.info(
        "Group-aware split: %d train / %d test records (%d unique groups, "
        "0 group overlap confirmed)",
        len(train_idx),
        len(test_idx),
        n_groups,
    )

    result = (
        X[train_idx].tolist(),
        X[test_idx].tolist(),
        y[train_idx].tolist(),
        y[test_idx].tolist(),
    )
    if return_test_groups:
        return result + (g[test_idx].tolist(),)
    return result


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def train_sentiment_classifier() -> dict:
    """
    Train the sentiment classifier and save artifacts.

    Returns a results dict with class counts, split sizes, and eval metrics.
    """
    log.info("=== SENTIMENT CLASSIFIER TRAINING ===")
    texts, labels, groups = load_sentiment_data()

    if len(texts) == 0:
        raise RuntimeError("No usable records for sentiment training.")

    dist = Counter(labels)
    log.info("Class distribution: %s", dict(dist))

    X_train, X_test, y_train, y_test = _group_split(texts, labels, groups)
    log.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    pipeline = _build_pipeline(class_weight="balanced")
    log.info("Fitting sentiment pipeline ...")
    pipeline.fit(X_train, y_train)

    # Evaluate (record-level only; no reviewer grouping for sentiment).
    metrics = evaluate_classifier(
        pipeline,
        X_test,
        y_test,
        classifier_name="Sentiment",
        label_names=sorted(dist.keys()),
    )

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "sentiment_pipeline.joblib"
    joblib.dump(pipeline, artifact_path)
    log.info("Sentiment pipeline saved -> %s", artifact_path)

    return {
        "classifier": "sentiment",
        "total_records": len(texts),
        "class_distribution": dict(dist),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "artifact": str(artifact_path),
        "metrics": metrics,
        "label_note": (
            "Labels are RATING-DERIVED: 1-2 stars -> Negative, 4-5 stars -> Positive. "
            "3-star reviews excluded as ambiguous. Not human-annotated sentiment."
        ),
    }


def train_authenticity_classifier() -> dict:
    """
    Train the authenticity classifier and save artifacts.

    Returns a results dict with class counts, split sizes, and eval metrics.
    """
    log.info("=== AUTHENTICITY CLASSIFIER TRAINING ===")
    texts, labels, groups = load_authenticity_data()

    if len(texts) == 0:
        raise RuntimeError("No usable records for authenticity training.")

    dist = Counter(labels)
    log.info("Class distribution: %s", dict(dist))

    min_class = min(dist.values())
    if min_class < 100:
        log.warning(
            "Smallest class has only %d examples. "
            "Model performance for that class may be unreliable.",
            min_class,
        )

    # Request test-group IDs so reviewer-level metrics can be computed.
    X_train, X_test, y_train, y_test, groups_test = _group_split(
        texts, labels, groups, return_test_groups=True
    )
    log.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    pipeline = _build_pipeline(class_weight="balanced")
    log.info("Fitting authenticity pipeline ...")
    pipeline.fit(X_train, y_train)

    # Evaluate — groups_test is passed to trigger reviewer-level metrics.
    # This surfaces that prolific reviewers inflate record-level accuracy.
    metrics = evaluate_classifier(
        pipeline,
        X_test,
        y_test,
        classifier_name="Authenticity",
        label_names=sorted(dist.keys()),
        groups=groups_test if groups_test else None,
    )

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "authenticity_pipeline.joblib"
    joblib.dump(pipeline, artifact_path)
    log.info("Authenticity pipeline saved -> %s", artifact_path)

    return {
        "classifier": "authenticity",
        "total_records": len(texts),
        "class_distribution": dict(dist),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "artifact": str(artifact_path),
        "metrics": metrics,
        "label_note": (
            "Labels are REVIEWER-ASSOCIATED: reviewer_labeled_fake==1 -> FAKE, "
            "reviewer_labeled_honest==1 -> HONEST. Records with both=0 or null are "
            "excluded. Labels describe the reviewer profile, not individual review "
            "content. Split is grouped by reviewer_id to prevent data leakage."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_training() -> dict:
    """
    Train both classifiers and return a combined results dict.
    Can be called from app.py or run directly: python -m src.model.train
    """
    sentiment_results = train_sentiment_classifier()
    authenticity_results = train_authenticity_classifier()

    return {
        "sentiment": sentiment_results,
        "authenticity": authenticity_results,
    }


if __name__ == "__main__":
    import json as _json
    # Configure logging only when running this module directly.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    results = run_training()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(_json.dumps(results, indent=2, default=str))
