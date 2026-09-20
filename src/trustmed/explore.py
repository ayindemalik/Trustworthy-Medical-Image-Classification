"""A first look at the data: how many cells of each type, and what they look like."""
import numpy as np

from trustmed import config, plots
from trustmed.data import apply_shift, load_split, to_float


def explore(size: int, limit: int | None = None):
    tag = config.run_tag(size, limit)
    counts = {}
    for split in ("train", "val", "test"):
        _, labels = load_split(split, size, limit)
        counts[split] = np.bincount(labels.numpy(), minlength=config.NUM_CLASSES)

    print(f"{'class':<22}{'train':>7}{'val':>7}{'test':>7}{'share':>8}")
    for c, name in enumerate(config.CLASS_NAMES):
        share = counts["train"][c] / counts["train"].sum()
        print(f"{name:<22}{counts['train'][c]:>7}{counts['val'][c]:>7}{counts['test'][c]:>7}{share:>8.1%}")
    print(f"{'total':<22}{counts['train'].sum():>7}{counts['val'].sum():>7}{counts['test'].sum():>7}")

    images, labels = load_split("train", size, limit)
    plots.samples(images, labels, f"samples_{tag}.png")

    # One neutrophil under every simulated shift, to see what the model will face
    first = int((labels == config.CLASS_NAMES.index("neutrophil")).nonzero()[0, 0])
    cell = to_float(images[first:first + 1], "cpu")
    shifted = {kind: [apply_shift(cell, kind, s)[0] for s in config.SEVERITIES] for kind in config.SHIFTS}
    plots.shift_examples(cell[0], shifted, f"shift_examples_{tag}.png")
