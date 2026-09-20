"""Every tunable value in one place.

Change a number here, not inside the code that uses it. A reader can then see
every choice the results depend on without searching the whole project.
"""
from pathlib import Path

# Folders, relative to the project root (where you run the commands)
DATA_DIR = Path("data")        # BloodMNIST .npz files (git-ignored)
MODELS_DIR = Path("models")    # trained weights (git-ignored)
OUTPUTS_DIR = Path("outputs")  # saved predictions (git-ignored)
RESULTS_DIR = Path("results")  # metric files (committed)
FIGURES_DIR = Path("figures")  # charts for the README (committed)

# Data
CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "immature granulocyte",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 64                # laptop default; Colab uses --size 224

# Training
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4                   # stop when validation accuracy has not improved for this many epochs
DROPOUT = 0.3                  # dropout before the last layer, reused by MC dropout

# Uncertainty
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]
MC_PASSES = 30
ECE_BINS = 15
COVERAGE = 0.90                # keep the 90% most confident cases, refer 10% to an expert

# Simulated distribution shift
SHIFTS = ["stain", "blur", "noise"]
SEVERITIES = [1, 2, 3, 4, 5]


def run_tag(size: int, limit: int | None = None) -> str:
    """Name used in every output file: '224px', or '64px_limit512' for a quick test."""
    return f"{size}px" if limit is None else f"{size}px_limit{limit}"
