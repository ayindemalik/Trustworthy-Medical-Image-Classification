"""Check the metrics on tiny cases where the right answer is known in advance."""
import numpy as np

from trustmed.calibration import ece, fit_temperature, nll, softmax
from trustmed.uncertainty import accuracy_at_coverage, aurc, risk_coverage


def test_softmax_rows_sum_to_one():
    probs = softmax(np.random.default_rng(0).normal(size=(5, 8)) * 10)
    assert np.allclose(probs.sum(axis=1), 1)


def test_ece_is_zero_when_confidence_matches_accuracy():
    # Ten answers at 80% confidence, eight of them right: perfectly honest
    probs = np.tile([0.8, 0.2], (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert ece(probs, labels) < 1e-9


def test_ece_catches_overconfidence():
    # 99% confident but right only half the time: the gap is 0.49
    probs = np.tile([0.99, 0.01], (10, 1))
    labels = np.array([0] * 5 + [1] * 5)
    assert abs(ece(probs, labels) - 0.49) < 1e-9


def test_temperature_scaling_never_changes_the_answer():
    logits = np.random.default_rng(1).normal(size=(200, 8)) * 5
    labels = np.random.default_rng(2).integers(0, 8, size=200)
    t = fit_temperature(logits, labels)
    assert np.array_equal(softmax(logits).argmax(1), softmax(logits / t).argmax(1))


def test_temperature_cools_down_an_overconfident_model():
    # Build honest logits, then multiply them by 5 to make the model overconfident
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(2000, 8))
    labels = np.array([rng.choice(8, p=p) for p in softmax(logits)])
    t = fit_temperature(logits * 5, labels)
    assert 4 < t < 6
    assert nll(softmax(logits * 5 / t), labels) < nll(softmax(logits * 5), labels)


def test_perfect_ranking_keeps_all_errors_for_last():
    # The two wrong answers are the two least confident: no errors until they are kept
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]])
    labels = np.array([0, 0, 0, 1, 1])
    coverage, risk = risk_coverage(probs, labels)
    assert np.allclose(risk[:3], 0)
    assert accuracy_at_coverage(probs, labels, coverage=0.6) == 1.0
    assert aurc(probs, labels) < aurc(probs[::-1].copy(), labels)   # worse when the order is reversed
