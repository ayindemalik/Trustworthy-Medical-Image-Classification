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

## Added at stage 5.1

def reliability(panels, filename):
    """One panel per method. Bars: accuracy per confidence group. Dashed line: perfectly honest confidence.

    `panels` maps a method name to (accuracy of each confidence group, ECE).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.6), sharey=True)
    centers = (np.arange(config.ECE_BINS) + 0.5) / config.ECE_BINS
    for ax, (name, (accuracy, ece)) in zip(axes, panels.items()):
        ax.bar(centers, np.nan_to_num(accuracy), width=0.9 / config.ECE_BINS, color=METHOD_COLORS[name])
        ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2)
        ax.set_title(f"{name}\nECE {ece:.3f}")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    _save(fig, filename)


def risk_coverage(curves, filename):
    """Error rate on the cases the model keeps, as it refers more of them to an expert.

    `curves` maps a method name to (coverage, risk), as returned by uncertainty.risk_coverage.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, (coverage, risk) in curves.items():
        shown = coverage >= 0.5   # the region that matters: referring up to half the cases
        ax.plot(coverage[shown], risk[shown] * 100, color=METHOD_COLORS[name], label=name)
    ax.axvline(config.COVERAGE, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0.5, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage: share of cases the model keeps")
    ax.set_ylabel("Error rate on kept cases (%)")
    ax.legend(loc="upper left")
    _save(fig, filename)


def shift(results, filename):
    """Accuracy and ECE as each kind of shift gets stronger. Severity 0 is the clean test set."""
    methods = list(results["conditions"]["clean"])
    severities = [0] + config.SEVERITIES
    fig, axes = plt.subplots(2, len(config.SHIFTS), figsize=(3.8 * len(config.SHIFTS), 6),
                             sharex=True, sharey="row", squeeze=False)
    for col, kind in enumerate(config.SHIFTS):
        names = ["clean"] + [f"{kind}{s}" for s in config.SEVERITIES]
        for row, (metric, label, scale) in enumerate([("accuracy", "Accuracy (%)", 100), ("ece", "ECE", 1)]):
            ax = axes[row, col]
            for method in methods:
                values = [results["conditions"][n][method][metric] * scale for n in names]
                ax.plot(severities, values, marker="o", color=METHOD_COLORS[method], label=method)
            ax.set_xticks(severities)
            if row == 0:
                ax.set_title(kind.capitalize())
            else:
                ax.set_xlabel("Severity (0 = clean)")
            if col == 0:
                ax.set_ylabel(label)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    _save(fig, filename)
