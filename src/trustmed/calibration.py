"""Is the model's confidence honest? Measure it (ECE) and fix it (temperature scaling)."""
import numpy as np

from trustmed import config


def softmax(logits):
    """Logits -> probabilities that sum to 1. Computed in float64 so tiny differences survive."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)   # subtract the max first so exp() cannot overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs, labels):
    """Negative log-likelihood: the average surprise at the true answer. Confident mistakes cost the most."""
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12)))


def brier(probs, labels):
    """Mean squared distance between the probabilities and the true answer written as 0s and a 1."""
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_bins(probs, labels, n_bins=config.ECE_BINS):
    """Put predictions into groups by confidence. Per group: mean confidence, accuracy, count."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    group = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    count = np.bincount(group, minlength=n_bins)
    with np.errstate(invalid="ignore"):   # empty groups give 0/0 = nan, which is fine
        mean_confidence = np.bincount(group, weights=confidence, minlength=n_bins) / count
        accuracy = np.bincount(group, weights=correct, minlength=n_bins) / count
    return mean_confidence, accuracy, count


def ece(probs, labels, n_bins=config.ECE_BINS):
    """Expected calibration error: the gap between confidence and accuracy, averaged over groups by size."""
    mean_confidence, accuracy, count = reliability_bins(probs, labels, n_bins)
    used = count > 0
    weight = count[used] / count.sum()
    return float(np.sum(weight * np.abs(accuracy[used] - mean_confidence[used])))


def fit_temperature(val_logits, val_labels):
    """Try 500 temperatures; keep the one with the lowest NLL on the validation split."""
    candidates = np.exp(np.linspace(np.log(0.05), np.log(20), 500))
    losses = [nll(softmax(val_logits / t), val_labels) for t in candidates]
    return float(candidates[int(np.argmin(losses))])
