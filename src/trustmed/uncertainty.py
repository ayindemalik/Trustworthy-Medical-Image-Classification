"""Uncertainty scores and selective prediction: keep the sure cases, refer the rest to an expert."""
import numpy as np
from sklearn.metrics import roc_auc_score

from trustmed import config


def entropy(probs):
    """0 when all belief is on one class; log(8) = 2.08 when it is spread evenly over all eight."""
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def risk_coverage(probs, labels):
    """Keep cases from most to least confident. For each count kept: coverage, and error rate among them."""
    order = np.argsort(-probs.max(axis=1), kind="stable")
    wrong = (probs.argmax(axis=1) != labels)[order]
    kept = np.arange(1, len(labels) + 1)
    return kept / len(labels), np.cumsum(wrong) / kept


def aurc(probs, labels):
    """Area under the risk-coverage curve: the error rate averaged over every cut-off. Lower is better."""
    _, risk = risk_coverage(probs, labels)
    return float(risk.mean())


def accuracy_at_coverage(probs, labels, coverage=config.COVERAGE):
    """Accuracy on the most confident share of cases; the rest are referred to an expert."""
    _, risk = risk_coverage(probs, labels)
    keep = max(1, int(np.ceil(coverage * len(labels))))
    return float(1 - risk[keep - 1])


def error_detection_auroc(probs, labels):
    """How well low confidence picks out the wrong answers. 0.5 = coin flip, 1.0 = perfect."""
    wrong = probs.argmax(axis=1) != labels
    if wrong.all() or not wrong.any():
        return float("nan")   # undefined when every answer is right, or every answer is wrong
    return float(roc_auc_score(wrong, -probs.max(axis=1)))
