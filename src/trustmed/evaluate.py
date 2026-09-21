"""Turn saved predictions into the metrics, tables and charts for the README."""
import json

import numpy as np
from sklearn.metrics import f1_score

from trustmed import config, plots
from trustmed.calibration import brier, ece, fit_temperature, nll, reliability_bins, softmax
from trustmed.predict import conditions
from trustmed.uncertainty import accuracy_at_coverage, aurc, entropy, error_detection_auroc, risk_coverage


def method_probs(saved, condition, temperature):
    """The methods compared. All of them start from the same trained networks."""
    logits = saved[f"{condition}_logits"]   # [models, N, 8]
    methods = {
        "Softmax": softmax(logits[0]),
        "Temperature scaling": softmax(logits[0] / temperature),
        "MC dropout": saved[f"{condition}_mc"],
    }
    if len(logits) > 1:
        methods["Deep ensemble"] = softmax(logits).mean(axis=0)   # average the models' probabilities
    return methods


def score(probs, labels):
    predicted = probs.argmax(axis=1)
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "ece": ece(probs, labels),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "aurc": aurc(probs, labels),
        "accuracy_at_coverage": accuracy_at_coverage(probs, labels),
        "error_auroc": error_detection_auroc(probs, labels),
        "mean_entropy": float(entropy(probs).mean()),
    }


def markdown_tables(results):
    """The tables, ready to paste into the README."""
    clean = results["conditions"]["clean"]
    methods = list(clean)
    lines = [
        f"# Results: {results['tag']}",
        "",
        f"{results['test_images']} test images, {results['models']} trained model(s). "
        f"Temperature fitted on the validation split: T = {results['temperature']:.3f}.",
        "",
        "## Methods on the clean test set",
        "",
        f"| Method | Accuracy | Macro-F1 | ECE | NLL | Brier | AURC | Accuracy at {results['coverage']:.0%} coverage | Error AUROC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for method, m in clean.items():
        lines.append(f"| {method} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['ece']:.4f} | {m['nll']:.4f} "
                     f"| {m['brier']:.4f} | {m['aurc']:.4f} | {m['accuracy_at_coverage']:.4f} | {m['error_auroc']:.4f} |")
    for metric, title in [("accuracy", "Accuracy"), ("ece", "ECE"), ("mean_entropy", "Mean entropy (uncertainty)")]:
        lines += ["", f"## Under shift: {title}", "",
                  "| Condition | " + " | ".join(methods) + " |",
                  "|---|" + "---|" * len(methods)]
        for name, _, _ in conditions():
            cells = " | ".join(f"{results['conditions'][name][m][metric]:.4f}" for m in methods)
            lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


def evaluate(size: int, limit: int | None = None):
    tag = config.run_tag(size, limit)
    path = config.OUTPUTS_DIR / f"predictions_{tag}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `trustmed predict` first.")
    saved = dict(np.load(path))
    labels = saved["test_labels"]

    # Fitted on the validation split only. Fitting on the test split would be cheating.
    temperature = fit_temperature(saved["val_logits"][0], saved["val_labels"])

    results = {"tag": tag, "test_images": int(len(labels)), "models": int(len(saved["clean_logits"])),
               "temperature": temperature, "coverage": config.COVERAGE, "conditions": {}}
    for name, _, _ in conditions():
        methods = method_probs(saved, name, temperature)
        results["conditions"][name] = {method: score(probs, labels) for method, probs in methods.items()}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(results, indent=2))
    tables = markdown_tables(results)
    (config.RESULTS_DIR / f"tables_{tag}.md").write_text(tables, encoding="utf-8")
    print(tables)
    print(f"Saved {config.RESULTS_DIR / f'metrics_{tag}.json'} and {config.RESULTS_DIR / f'tables_{tag}.md'}")

    clean = method_probs(saved, "clean", temperature)
    plots.reliability({name: (reliability_bins(p, labels)[1], ece(p, labels)) for name, p in clean.items()},
                      f"reliability_{tag}.png")
    plots.risk_coverage({name: risk_coverage(p, labels) for name, p in clean.items()}, f"risk_coverage_{tag}.png")
    plots.shift(results, f"shift_{tag}.png")
