"""Every chart in the project, in one shared style, so the README figures read as one set."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from trustmed import config

plt.switch_backend("Agg")   # draw straight to image files; no window needed (works on Colab too)

# One fixed colour per method, always in this order (a palette checked for colour blindness)
METHOD_COLORS = {
    "Softmax": "#2a78d6",
    "Temperature scaling": "#eb6834",
    "MC dropout": "#1baf7a",
    "Deep ensemble": "#eda100",
}
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HEAT = "#e34948"   # Grad-CAM colour, drawn over a grey copy of the cell

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False,
})


def _save(fig, filename):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _bare(ax):
    """A picture panel: no ticks, no grid, no frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def samples(images, labels, filename, per_class=6):
    """A grid of example cells, one row per class."""
    fig, axes = plt.subplots(config.NUM_CLASSES, per_class, figsize=(per_class * 1.1 + 1.8, config.NUM_CLASSES * 1.1))
    for c, name in enumerate(config.CLASS_NAMES):
        chosen = (labels == c).nonzero()[:per_class, 0]
        for j, ax in enumerate(axes[c]):
            ax.axis("off")
            if j < len(chosen):
                ax.imshow(images[chosen[j]].numpy())
        axes[c, 0].text(-0.1, 0.5, name, transform=axes[c, 0].transAxes, ha="right", va="center")
    _save(fig, filename)


def shift_examples(clean, shifted, filename):
    """One cell under every kind and strength of simulated shift."""
    columns = len(config.SEVERITIES) + 1
    fig, axes = plt.subplots(len(shifted), columns, figsize=(columns * 1.3 + 0.8, len(shifted) * 1.4))
    for r, (kind, images) in enumerate(shifted.items()):
        for c, image in enumerate([clean] + images):
            ax = axes[r, c]
            ax.imshow(image.permute(1, 2, 0).numpy())
            _bare(ax)
            if r == 0:
                ax.set_title("clean" if c == 0 else f"severity {c}", fontsize=9)
        axes[r, 0].set_ylabel(kind)
    _save(fig, filename)
