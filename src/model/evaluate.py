"""
GhostSignal — ML Evaluation
==============================
Computes and prints evaluation metrics for a trained sklearn pipeline.

Metrics reported:
  - Accuracy
  - Per-class Precision, Recall, F1
  - Macro-averaged Precision, Recall, F1 (appropriate for imbalanced classes)
  - Weighted-averaged F1
  - Confusion matrix
  - Full sklearn classification report

This module has no side effects beyond printing to stdout/logger.
It does not modify any files or datasets.
"""

import logging
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

log = logging.getLogger(__name__)


def evaluate_classifier(
    pipeline,
    X_test: list[str],
    y_test: list[str],
    classifier_name: str = "Classifier",
    label_names: list[str] | None = None,
    groups: list[str] | None = None,
) -> dict:
    """
    Evaluate a fitted sklearn pipeline on a held-out test set.

    Parameters
    ----------
    pipeline        : fitted sklearn Pipeline (must have predict / predict_proba)
    X_test          : list of raw text strings
    y_test          : list of ground-truth label strings
    classifier_name : human-readable name for logging
    label_names     : ordered list of class names for the confusion matrix
    groups          : optional list of group IDs (e.g. reviewer_id) aligned with
                      X_test/y_test.  When supplied, reviewer-level metrics are
                      computed alongside record-level metrics.  For each group the
                      majority-vote predicted label is compared to the group's true
                      label (all records in a group share one label by construction).
                      This gives a fairer view of classifier quality because prolific
                      reviewers cannot inflate record-level counts.

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        f1_weighted, confusion_matrix, classification_report,
        test_size, class_distribution,
        and optionally reviewer_level_metrics (when groups is provided)
    """
    if len(X_test) == 0:
        log.warning("[%s] Empty test set — skipping evaluation.", classifier_name)
        return {"error": "empty_test_set"}

    y_pred = pipeline.predict(X_test)

    accuracy        = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro        = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=label_names)
    report_str = classification_report(
        y_test, y_pred, labels=label_names, zero_division=0
    )

    test_dist = dict(Counter(y_test))

    # --- Print to stdout so it's visible during training runs ---
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  EVALUATION: {classifier_name}")
    print(sep)
    print(f"  Test-set size     : {len(X_test):,}")
    print(f"  Class distribution: {test_dist}")
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  Precision (macro) : {precision_macro:.4f}")
    print(f"  Recall    (macro) : {recall_macro:.4f}")
    print(f"  F1        (macro) : {f1_macro:.4f}")
    print(f"  F1     (weighted) : {f1_weighted:.4f}")
    print()
    print("  Confusion matrix:")
    if label_names:
        col_header = "  " + "".join(f"{lbl:>12}" for lbl in label_names)
        print(col_header)
        for i, row_lbl in enumerate(label_names):
            row = "  " + f"{row_lbl:<10}" + "".join(f"{v:>12,}" for v in cm[i])
            print(row)
    else:
        print(cm)
    print()
    print("  Classification report:")
    for line in report_str.splitlines():
        print("  " + line)
    print(sep)

    result = {
        "test_size": len(X_test),
        "class_distribution": test_dist,
        "accuracy": round(accuracy, 6),
        "precision_macro": round(precision_macro, 6),
        "recall_macro": round(recall_macro, 6),
        "f1_macro": round(f1_macro, 6),
        "f1_weighted": round(f1_weighted, 6),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_names or sorted(set(y_test)),
        "classification_report": report_str,
    }

    # ------------------------------------------------------------------
    # Optional reviewer-level metrics
    # ------------------------------------------------------------------
    # When groups are provided (reviewer_id per test record), aggregate
    # predictions per group using majority vote, then compute metrics at
    # the reviewer level.  This gives a fairer view when prolific reviewers
    # contribute many test records with the same label, which would otherwise
    # inflate record-level accuracy.
    #
    # Aggregation rule: the majority-vote predicted label for a group is the
    # most common prediction across all of that group's test records.  Ties
    # are broken by taking the first label alphabetically, which is consistent
    # and deterministic.  The true label for each group is taken from the first
    # record (all records in a group share the same true label by construction).
    if groups is not None and len(groups) == len(X_test):
        group_true: dict[str, str] = {}
        group_preds: dict[str, list[str]] = {}

        for grp, true_lbl, pred_lbl in zip(groups, y_test, y_pred):
            if grp not in group_true:
                group_true[grp] = true_lbl
            group_preds.setdefault(grp, []).append(pred_lbl)

        reviewer_y_true = []
        reviewer_y_pred = []
        for grp, true_lbl in group_true.items():
            preds = group_preds[grp]
            pred_counts = Counter(preds)
            # majority vote; tie-break alphabetically
            maj_vote = max(pred_counts, key=lambda x: (pred_counts[x], x))
            reviewer_y_true.append(true_lbl)
            reviewer_y_pred.append(maj_vote)

        rv_accuracy = accuracy_score(reviewer_y_true, reviewer_y_pred)
        rv_precision = precision_score(
            reviewer_y_true, reviewer_y_pred, average="macro", zero_division=0
        )
        rv_recall = recall_score(
            reviewer_y_true, reviewer_y_pred, average="macro", zero_division=0
        )
        rv_f1_macro = f1_score(
            reviewer_y_true, reviewer_y_pred, average="macro", zero_division=0
        )
        rv_f1_weighted = f1_score(
            reviewer_y_true, reviewer_y_pred, average="weighted", zero_division=0
        )
        rv_cm = confusion_matrix(reviewer_y_true, reviewer_y_pred, labels=label_names)
        rv_report = classification_report(
            reviewer_y_true, reviewer_y_pred, labels=label_names, zero_division=0
        )
        rv_dist = dict(Counter(reviewer_y_true))

        print(f"\n{sep}")
        print(f"  REVIEWER-LEVEL EVALUATION: {classifier_name}")
        print(f"  (majority-vote per reviewer — one prediction per reviewer)")
        print(sep)
        print(f"  Unique reviewers in test  : {len(group_true):,}")
        print(f"  Reviewer class distribution: {rv_dist}")
        print(f"  Accuracy          : {rv_accuracy:.4f}")
        print(f"  Precision (macro) : {rv_precision:.4f}")
        print(f"  Recall    (macro) : {rv_recall:.4f}")
        print(f"  F1        (macro) : {rv_f1_macro:.4f}")
        print(f"  F1     (weighted) : {rv_f1_weighted:.4f}")
        print()
        print("  Confusion matrix (reviewer-level):")
        if label_names:
            col_header = "  " + "".join(f"{lbl:>12}" for lbl in label_names)
            print(col_header)
            for i, row_lbl in enumerate(label_names):
                row = "  " + f"{row_lbl:<10}" + "".join(f"{v:>12,}" for v in rv_cm[i])
                print(row)
        else:
            print(rv_cm)
        print()
        print("  Classification report (reviewer-level):")
        for line in rv_report.splitlines():
            print("  " + line)
        print(sep)

        result["reviewer_level_metrics"] = {
            "n_reviewers": len(group_true),
            "reviewer_class_distribution": rv_dist,
            "accuracy": round(rv_accuracy, 6),
            "precision_macro": round(rv_precision, 6),
            "recall_macro": round(rv_recall, 6),
            "f1_macro": round(rv_f1_macro, 6),
            "f1_weighted": round(rv_f1_weighted, 6),
            "confusion_matrix": rv_cm.tolist(),
            "confusion_matrix_labels": label_names or sorted(set(reviewer_y_true)),
            "classification_report": rv_report,
            "aggregation_method": "majority_vote_per_reviewer_id",
        }
    else:
        result["reviewer_level_metrics"] = None

    return result
