"""Run every trained model over the validation and test images, clean and shifted, and save it all.

This is the only heavy part of the evaluation. Everything after it (metrics,
tables, charts) reads the saved file, so it runs in seconds on any laptop.
"""
import time

import numpy as np
import torch

from trustmed import config
from trustmed.data import load_split
from trustmed.model import checkpoint_path, get_device, load, run_model


def conditions():
    """The clean test set, then every kind of shift at every severity."""
    yield "clean", None, 0
    for kind in config.SHIFTS:
        for severity in config.SEVERITIES:
            yield f"{kind}{severity}", kind, severity


def predict(size: int, limit: int | None = None):
    tag = config.run_tag(size, limit)
    device = get_device()
    paths = [checkpoint_path(tag, seed) for seed in config.ENSEMBLE_SEEDS]
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise SystemExit(f"No trained models for {tag} in {config.MODELS_DIR}/. Run `trustmed train` first.")

    val_x, val_y = load_split("val", size, limit)
    test_x, test_y = load_split("test", size, limit)
    torch.manual_seed(0)  # makes the MC-dropout passes repeatable

    logits = {"val": []} | {name: [] for name, _, _ in conditions()}
    saved = {"val_labels": val_y.numpy(), "test_labels": test_y.numpy()}
    for i, path in enumerate(paths):
        start = time.perf_counter()
        model, info = load(path, device)
        logits["val"].append(run_model(model, val_x, device)[0])
        for name, kind, severity in conditions():
            passes = config.MC_PASSES if i == 0 else 0   # MC dropout uses the first model only
            output, mc_probs = run_model(model, test_x, device, kind, severity, mc_passes=passes)
            logits[name].append(output)
            if mc_probs is not None:
                saved[f"{name}_mc"] = mc_probs.numpy()
        clean = (logits["clean"][-1].argmax(1) == test_y).float().mean().item()
        print(f"{path.name}: clean test accuracy {clean:.4f} "
              f"({time.perf_counter() - start:.0f}s for all {len(logits) - 1} test conditions)")

    for name, per_model in logits.items():
        saved[f"{name}_logits"] = torch.stack(per_model).numpy()   # [models, N, 8]
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUTS_DIR / f"predictions_{tag}.npz"
    np.savez_compressed(out, **saved)
    print(f"Saved predictions of {len(paths)} model(s) to {out}")
