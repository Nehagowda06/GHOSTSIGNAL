"""
Tests for the GhostSignal ML module.

Test organisation
-----------------
Unit tests (no model artifacts required)
  - Use lightweight mock pipelines so they run on any machine, including CI,
    before train.py has been executed.
  - Marked with the 'unit' pytest mark.

Integration tests (require saved model artifacts in models/)
  - Load the real trained pipelines from disk.
  - Skipped automatically if the artifact files are absent.
  - Marked with the 'integration' pytest mark.

Run all:          python -m pytest tests/test_classifier.py -v
Run unit only:    python -m pytest tests/test_classifier.py -v -m unit
Run integration:  python -m pytest tests/test_classifier.py -v -m integration
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.classifier import (
    AUTHENTICITY_PATH,
    SENTIMENT_PATH,
    GhostSignalClassifier,
    PredictionResult,
    SinglePrediction,
)

# ---------------------------------------------------------------------------
# Skip guard for integration tests
# ---------------------------------------------------------------------------

_ARTIFACTS_AVAILABLE = SENTIMENT_PATH.exists() and AUTHENTICITY_PATH.exists()
_SKIP_INTEGRATION = pytest.mark.skipif(
    not _ARTIFACTS_AVAILABLE,
    reason=(
        "Trained model artifacts not found in models/. "
        "Run 'python -m src.model.train' to generate them."
    ),
)


# ---------------------------------------------------------------------------
# Helpers — build a minimal mock sklearn pipeline
# ---------------------------------------------------------------------------

def _make_mock_pipeline(classes: list[str], proba_values: list[float]) -> Any:
    """
    Return a MagicMock that quacks like a fitted sklearn Pipeline.

    Parameters
    ----------
    classes      : class labels, e.g. ['Negative', 'Positive']
    proba_values : probability for each class (must sum to 1.0)
    """
    assert len(classes) == len(proba_values)
    pipeline = MagicMock()
    pipeline.classes_ = np.array(classes)
    pipeline.predict.return_value = np.array([classes[int(np.argmax(proba_values))]])
    pipeline.predict_proba.return_value = np.array([proba_values])
    return pipeline


def _make_classifier_with_mocks(
    sent_classes: list[str],
    sent_proba: list[float],
    auth_classes: list[str],
    auth_proba: list[float],
) -> GhostSignalClassifier:
    """
    Return a GhostSignalClassifier whose pipelines are pre-loaded mock objects.
    No files are read from disk.
    """
    clf = GhostSignalClassifier()
    clf._sentiment_pipeline = _make_mock_pipeline(sent_classes, sent_proba)
    clf._authenticity_pipeline = _make_mock_pipeline(auth_classes, auth_proba)
    clf._sentiment_loaded = True
    clf._authenticity_loaded = True
    return clf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_clf():
    """
    Module-scoped classifier backed by mock pipelines.
    Safe to use in all unit tests — no disk I/O.
    """
    return _make_classifier_with_mocks(
        sent_classes=["Negative", "Positive"],
        sent_proba=[0.05, 0.95],
        auth_classes=["FAKE", "HONEST"],
        auth_proba=[0.20, 0.80],
    )


@pytest.fixture(scope="module")
def real_clf():
    """
    Module-scoped classifier that loads real trained pipelines from disk.
    Only used in integration tests; skipped when artifacts are absent.
    """
    return GhostSignalClassifier()


# ===========================================================================
# UNIT TESTS — no artifacts required
# ===========================================================================

# ---------------------------------------------------------------------------
# Return types and structure
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_predict_returns_prediction_result(mock_clf):
    result = mock_clf.predict("Great product, very happy with it.")
    assert isinstance(result, PredictionResult)


@pytest.mark.unit
def test_prediction_result_has_correct_fields(mock_clf):
    result = mock_clf.predict("Terrible quality. Broke in two days.")
    assert isinstance(result.sentiment, SinglePrediction)
    assert isinstance(result.authenticity, SinglePrediction)
    assert isinstance(result.text_preview, str)
    assert len(result.text_preview) > 0


@pytest.mark.unit
def test_probabilities_are_dict_with_str_keys_and_float_values(mock_clf):
    result = mock_clf.predict("Really loved this product.")
    for pred in (result.sentiment, result.authenticity):
        assert isinstance(pred.probabilities, dict)
        assert len(pred.probabilities) > 0
        for k, v in pred.probabilities.items():
            assert type(k) is str, f"Key should be plain str, got {type(k)}"
            assert type(v) is float, f"Value should be float, got {type(v)}"
            assert 0.0 <= v <= 1.0


@pytest.mark.unit
def test_probabilities_sum_to_one(mock_clf):
    result = mock_clf.predict("Average product, nothing special.")
    for pred in (result.sentiment, result.authenticity):
        total = sum(pred.probabilities.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"


# ---------------------------------------------------------------------------
# Uncertainty threshold
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_high_confidence_prediction_not_uncertain(mock_clf):
    """Mock returns 0.95 for Positive — well above 0.60."""
    result = mock_clf.predict("Amazing product!")
    assert result.sentiment.is_uncertain is False
    assert result.sentiment.label == "Positive"


@pytest.mark.unit
def test_low_confidence_prediction_marked_uncertain():
    """Mock returns 0.55 max probability — below the 0.60 threshold."""
    clf = _make_classifier_with_mocks(
        sent_classes=["Negative", "Positive"],
        sent_proba=[0.45, 0.55],      # max = 0.55 < 0.60
        auth_classes=["FAKE", "HONEST"],
        auth_proba=[0.48, 0.52],      # max = 0.52 < 0.60
    )
    result = clf.predict("Some review text.")
    assert result.sentiment.is_uncertain is True
    assert result.sentiment.label == "uncertain"
    assert result.authenticity.is_uncertain is True
    assert result.authenticity.label == "uncertain"


@pytest.mark.unit
def test_exactly_at_threshold_is_uncertain():
    """A max probability of exactly 0.60 is NOT below the threshold (< not <=)."""
    clf = _make_classifier_with_mocks(
        sent_classes=["Negative", "Positive"],
        sent_proba=[0.40, 0.60],      # exactly 0.60 — should NOT be uncertain
        auth_classes=["FAKE", "HONEST"],
        auth_proba=[0.40, 0.60],
    )
    result = clf.predict("Some review text.")
    assert result.sentiment.is_uncertain is False
    assert result.sentiment.label == "Positive"


# ---------------------------------------------------------------------------
# Edge case inputs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_string_returns_unavailable(mock_clf):
    result = mock_clf.predict("")
    assert result.sentiment.label == "unavailable"
    assert result.authenticity.label == "unavailable"
    assert result.sentiment.error is not None
    assert result.authenticity.error is not None


@pytest.mark.unit
def test_whitespace_only_returns_unavailable(mock_clf):
    result = mock_clf.predict("   \n\t  ")
    assert result.sentiment.label == "unavailable"
    assert result.authenticity.label == "unavailable"


@pytest.mark.unit
def test_long_text_preview_truncated_to_123_chars(mock_clf):
    text = "x" * 200
    result = mock_clf.predict(text)
    assert result.text_preview.endswith("...")
    assert len(result.text_preview) == 123  # 120 chars + "..."


@pytest.mark.unit
def test_short_text_preview_not_truncated(mock_clf):
    text = "Good product."
    result = mock_clf.predict(text)
    assert result.text_preview == text
    assert not result.text_preview.endswith("...")


@pytest.mark.unit
def test_long_text_preview_at_most_123_chars(mock_clf):
    long_text = "great product " * 50
    result = mock_clf.predict(long_text)
    assert len(result.text_preview) <= 123


# ---------------------------------------------------------------------------
# Missing model handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_missing_sentiment_model_returns_unavailable_for_sentiment():
    """When the sentiment pipeline is not loaded, that result is 'unavailable'
    but authenticity still works."""
    clf = GhostSignalClassifier()
    clf._sentiment_pipeline = None
    clf._sentiment_loaded = False
    clf._authenticity_pipeline = _make_mock_pipeline(["FAKE", "HONEST"], [0.3, 0.7])
    clf._authenticity_loaded = True

    result = clf.predict("Some review text.")
    assert result.sentiment.model_available is False
    assert result.sentiment.label == "unavailable"
    assert result.authenticity.model_available is True
    assert result.authenticity.label != "unavailable"


@pytest.mark.unit
def test_missing_authenticity_model_returns_unavailable_for_authenticity():
    clf = GhostSignalClassifier()
    clf._sentiment_pipeline = _make_mock_pipeline(["Negative", "Positive"], [0.1, 0.9])
    clf._sentiment_loaded = True
    clf._authenticity_pipeline = None
    clf._authenticity_loaded = False

    result = clf.predict("Some review text.")
    assert result.authenticity.model_available is False
    assert result.authenticity.label == "unavailable"
    assert result.sentiment.model_available is True
    assert result.sentiment.label != "unavailable"


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_batch_predict_returns_list(mock_clf):
    texts = ["Love this!", "Hate this!", "It is okay."]
    results = mock_clf.predict_batch(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)
    for r in results:
        assert isinstance(r, PredictionResult)


@pytest.mark.unit
def test_batch_predict_empty_list(mock_clf):
    assert mock_clf.predict_batch([]) == []


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prediction_result_is_json_serialisable(mock_clf):
    import json
    from dataclasses import asdict
    result = mock_clf.predict("Really great product.")
    d = asdict(result)
    serialised = json.dumps(d)   # must not raise
    assert isinstance(serialised, str)
    assert len(serialised) > 0


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_repr_shows_loaded_when_loaded(mock_clf):
    """repr should report 'loaded' (not 'not loaded') when both models are set."""
    r = repr(mock_clf)
    assert "sentiment=loaded" in r, f"Expected 'sentiment=loaded' in repr, got: {r}"
    assert "authenticity=loaded" in r, f"Expected 'authenticity=loaded' in repr, got: {r}"


@pytest.mark.unit
def test_repr_shows_not_loaded_when_not_loaded():
    """repr should report 'not loaded' for a fresh instance with no pipelines."""
    clf = GhostSignalClassifier()
    # Do NOT call is_ready() — keep the pipelines unloaded.
    r = repr(clf)
    assert "sentiment=not loaded" in r, f"Expected 'not loaded', got: {r}"
    assert "authenticity=not loaded" in r, f"Expected 'not loaded', got: {r}"


# ---------------------------------------------------------------------------
# _group_split fallback path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_group_split_raises_on_gss_failure():
    """
    When GroupShuffleSplit cannot produce a valid split, _group_split must
    raise RuntimeError.  It must never fall back to a record-level split,
    because that would allow the same reviewer/author to appear in both
    train and test sets.
    """
    import unittest.mock as mock
    from src.model.train import _group_split

    texts  = [f"review text {i}" for i in range(10)]
    labels = ["FAKE", "HONEST"] * 5
    groups = [f"reviewer_{i}" for i in range(10)]

    with mock.patch("src.model.train.GroupShuffleSplit") as mock_gss:
        mock_gss.return_value.split.side_effect = ValueError("Cannot split")
        with pytest.raises(RuntimeError) as exc_info:
            _group_split(texts, labels, groups, test_size=0.3)

    msg = str(exc_info.value)
    # The error must name the root cause
    assert "Cannot split" in msg, f"Original error not surfaced in message: {msg}"
    # The error must explain WHY there is no fallback
    assert "leakage" in msg.lower() or "reviewer" in msg.lower(), (
        f"Message does not explain the leakage risk: {msg}"
    )


@pytest.mark.unit
def test_group_split_raises_on_gss_failure_with_return_test_groups():
    """
    RuntimeError is raised regardless of the return_test_groups flag.
    There is no silent record-level fallback in either code path.
    """
    import unittest.mock as mock
    from src.model.train import _group_split

    texts  = [f"review text {i}" for i in range(10)]
    labels = ["FAKE", "HONEST"] * 5
    groups = [f"reviewer_{i}" for i in range(10)]

    with mock.patch("src.model.train.GroupShuffleSplit") as mock_gss:
        mock_gss.return_value.split.side_effect = ValueError("Cannot split")
        with pytest.raises(RuntimeError):
            _group_split(texts, labels, groups, test_size=0.3, return_test_groups=True)


@pytest.mark.unit
def test_group_split_no_record_level_fallback():
    """
    Confirm that _group_split does not import or call train_test_split.
    This is a defence-in-depth static check: even if the runtime tests above
    pass, we verify there is no callable fallback that could silently mix
    reviewer/author records across train and test.
    """
    import inspect
    from src.model import train as train_module
    from src.model.train import _group_split

    # train_test_split must not be importable from the train module namespace.
    assert not hasattr(train_module, "train_test_split"), (
        "train_test_split is importable from train.py — remove it to prevent "
        "accidental use as a fallback."
    )

    # The _group_split source must not contain a *call* to train_test_split.
    # We check for the call pattern "train_test_split(" rather than the bare
    # name, so that mentions in docstrings or comments do not trigger a false
    # positive.
    source = inspect.getsource(_group_split)
    assert "train_test_split(" not in source, (
        "train_test_split() is called inside _group_split — "
        "this would re-introduce the record-level fallback that allows "
        "reviewer/author leakage."
    )


# ===========================================================================
# INTEGRATION TESTS — require saved model artifacts
# ===========================================================================

@_SKIP_INTEGRATION
def test_integration_models_are_ready(real_clf):
    status = real_clf.is_ready()
    assert status["sentiment"] is True, "Sentiment pipeline failed to load."
    assert status["authenticity"] is True, "Authenticity pipeline failed to load."


@_SKIP_INTEGRATION
def test_integration_clearly_positive_review(real_clf):
    result = real_clf.predict("Absolutely love this product! Best purchase I have ever made.")
    assert result.sentiment.label == "Positive"
    assert result.sentiment.model_available is True


@_SKIP_INTEGRATION
def test_integration_clearly_negative_review(real_clf):
    result = real_clf.predict("Complete garbage. Stopped working immediately. Total waste of money.")
    assert result.sentiment.label == "Negative"
    assert result.sentiment.model_available is True


@_SKIP_INTEGRATION
def test_integration_sentiment_classes_are_valid(real_clf):
    for text in [
        "Great value for the price.",
        "Very disappointed with this purchase.",
        "It arrived on time.",
    ]:
        label = real_clf.predict(text).sentiment.label
        assert label in ("Positive", "Negative", "uncertain"), (
            f"Unexpected sentiment label: {label!r}"
        )


@_SKIP_INTEGRATION
def test_integration_authenticity_classes_are_valid(real_clf):
    for text in [
        "Great product, would recommend.",
        "Do not buy this, looks like a scam.",
        "Five stars, amazing quality.",
    ]:
        label = real_clf.predict(text).authenticity.label
        assert label in ("FAKE", "HONEST", "uncertain"), (
            f"Unexpected authenticity label: {label!r}"
        )


@_SKIP_INTEGRATION
def test_integration_repr_shows_loaded(real_clf):
    real_clf.is_ready()  # ensure both pipelines are loaded
    r = repr(real_clf)
    assert "sentiment=loaded" in r, f"Expected 'sentiment=loaded', got: {r}"
    assert "authenticity=loaded" in r, f"Expected 'authenticity=loaded', got: {r}"


@_SKIP_INTEGRATION
def test_integration_probabilities_sum_to_one(real_clf):
    result = real_clf.predict("Average product, nothing special.")
    for pred in (result.sentiment, result.authenticity):
        total = sum(pred.probabilities.values())
        assert abs(total - 1.0) < 0.01
