# Trustworthy Medical Image Classification

> A blood-cell classifier that reports how sure it is, hands its least certain cases to a human, and shows where in the image it looked.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Status:** 🚧 In active development. Every figure in this README comes from a committed run in `results/`. Until a run exists, the value reads _pending_.

🔗 **Live demo:** coming soon

<!-- Record the Gradio demo, save to assets/demo.gif, then uncomment.
![demo](assets/demo.gif)
-->

---

## Overview

**Problem.** A standard image classifier gives its wrong answers with the same high confidence as its right ones. At 96% accuracy, one cell in twenty-five is misclassified, and the softmax score does not say which one. In a hematology lab, where a blood-cell differential count decides what happens next for a patient, a model that cannot tell when it is unsure cannot safely take over any of the work.

**Approach.** A ResNet-18, pretrained on ImageNet and fine-tuned on BloodMNIST (eight classes of peripheral blood cell), with three layers of trust added on top:

1. **Calibration.** Does 90% confidence mean right 90% of the time? Measured with expected calibration error (ECE) and reliability diagrams. Corrected with temperature scaling, fitted on the validation split only.
2. **Uncertainty.** MC dropout on the classification head and a five-model deep ensemble, compared against the plain softmax. The uncertainty score drives **selective prediction**: the model keeps the cases it is sure about and refers the rest to an expert.
3. **Explanation.** Grad-CAM heatmaps of the image regions behind each prediction, plus a sanity check that the heatmaps depend on the trained weights and are not just edge detection.

All three are then re-measured on a **shifted test set**, with simulated staining, focus and noise changes, because a second microscope in a second lab is the realistic way these models fail.

**Why it matters.** A clinician does not ask how accurate a model is on average. They ask whether they can trust this particular answer. The headline result here is a risk–coverage curve, which shows how accurate the model is on the cases it chooses to keep, and how many it has to hand back to get there.

## Results

Evaluated on the official BloodMNIST test split (3,421 images).  Filled from `results/` as each step lands.

# Results: 224px

3421 test images, 5 trained model(s). Temperature fitted on the validation split: T = 1.134.

Trained on a Colab T4 GPU at 224 px. Temperature T = 1.134. Grad-CAM sanity check: Spearman = 0.2976 (trained vs random weights, 3421 test images).


| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | _pending_ | Single ResNet-18 |
| Macro-F1 | _pending_ | Weights all eight classes equally, so rare classes count |
| ECE (before → after temperature scaling) | _pending_ | 15 equal-width bins; lower is better |
| Accuracy at 90% coverage | _pending_ | Most uncertain 10% referred to an expert |
| AURC | _pending_ | Area under the risk–coverage curve; lower is better |

## Methods on the clean test set

| Method | Accuracy | Macro-F1 | ECE | NLL | Brier | AURC | Accuracy at 90% coverage | Error AUROC |
|---|---|---|---|---|---|---|---|---|
| Softmax | 0.9927 | 0.9940 | 0.0047 | 0.0331 | 0.0127 | 0.0013 | 0.9990 | 0.9165 |
| Temperature scaling | 0.9927 | 0.9940 | 0.0032 | 0.0313 | 0.0127 | 0.0013 | 0.9990 | 0.9167 |
| MC dropout | 0.9927 | 0.9940 | 0.0037 | 0.0325 | 0.0126 | 0.0013 | 0.9990 | 0.9165 |
| Deep ensemble | 0.9936 | 0.9948 | 0.0034 | 0.0312 | 0.0121 | 0.0013 | 0.9987 | 0.8985 |

### Uncertainty methods compared

| Method | Accuracy | ECE | NLL | AURC | Extra cost |
|--------|----------|-----|-----|------|------------|
| Softmax (baseline) | _pending_ | _pending_ | _pending_ | _pending_ | none |
| + Temperature scaling | _pending_ | _pending_ | _pending_ | _pending_ | one parameter |
| MC dropout (30 passes, head only) | _pending_ | _pending_ | _pending_ | _pending_ | last layer run 30× |
| Deep ensemble (5 models) | _pending_ | _pending_ | _pending_ | _pending_ | 5× training and inference |

# Under shift: Accuracy

| Condition | Softmax | Temperature scaling | MC dropout | Deep ensemble |
|---|---|---|---|---|
| clean | 0.9927 | 0.9927 | 0.9927 | 0.9936 |
| stain1 | 0.9763 | 0.9763 | 0.9754 | 0.9752 |
| stain2 | 0.7524 | 0.7524 | 0.7539 | 0.6840 |
| stain3 | 0.4364 | 0.4364 | 0.4358 | 0.4072 |
| stain4 | 0.2607 | 0.2607 | 0.2605 | 0.2455 |
| stain5 | 0.2014 | 0.2014 | 0.2014 | 0.1926 |
| blur1 | 0.9860 | 0.9860 | 0.9860 | 0.9863 |
| blur2 | 0.9398 | 0.9398 | 0.9395 | 0.9418 |
| blur3 | 0.8503 | 0.8503 | 0.8503 | 0.8585 |
| blur4 | 0.6977 | 0.6977 | 0.6998 | 0.6524 |
| blur5 | 0.5989 | 0.5989 | 0.6004 | 0.5063 |
| noise1 | 0.9828 | 0.9828 | 0.9830 | 0.9906 |
| noise2 | 0.9237 | 0.9237 | 0.9231 | 0.9588 |
| noise3 | 0.6106 | 0.6106 | 0.6115 | 0.7729 |
| noise4 | 0.5288 | 0.5288 | 0.5285 | 0.5519 |
| noise5 | 0.4382 | 0.4382 | 0.4396 | 0.4665 |

### Under distribution shift (deep ensemble)

| Shift | Severity | Accuracy | ECE | Mean entropy |
|-------|----------|----------|-----|--------------|
| None | – | _pending_ | _pending_ | _pending_ |
| Stain (hue / saturation) | 1 / 3 / 5 | _pending_ | _pending_ | _pending_ |
| Defocus blur | 1 / 3 / 5 | _pending_ | _pending_ | _pending_ |
| Sensor noise | 1 / 3 / 5 | _pending_ | _pending_ | _pending_ |

## Under shift: ECE

| Condition | Softmax | Temperature scaling | MC dropout | Deep ensemble |
|---|---|---|---|---|
| clean | 0.0047 | 0.0032 | 0.0037 | 0.0034 |
| stain1 | 0.0081 | 0.0062 | 0.0082 | 0.0187 |
| stain2 | 0.1770 | 0.1669 | 0.1704 | 0.2034 |
| stain3 | 0.4989 | 0.4891 | 0.4934 | 0.4894 |
| stain4 | 0.7020 | 0.6962 | 0.6987 | 0.6990 |
| stain5 | 0.7832 | 0.7804 | 0.7815 | 0.7847 |
| blur1 | 0.0073 | 0.0054 | 0.0060 | 0.0056 |
| blur2 | 0.0342 | 0.0291 | 0.0328 | 0.0132 |
| blur3 | 0.0733 | 0.0576 | 0.0702 | 0.0217 |
| blur4 | 0.1385 | 0.1118 | 0.1309 | 0.0612 |
| blur5 | 0.1313 | 0.0888 | 0.1273 | 0.1104 |
| noise1 | 0.0075 | 0.0060 | 0.0058 | 0.0077 |
| noise2 | 0.0392 | 0.0317 | 0.0365 | 0.0226 |
| noise3 | 0.2564 | 0.2414 | 0.2488 | 0.0789 |
| noise4 | 0.3562 | 0.3376 | 0.3435 | 0.2167 |
| noise5 | 0.4159 | 0.3959 | 0.3940 | 0.3018 |

## Under shift: Accuracy

| Condition | Softmax | Temperature scaling | MC dropout | Deep ensemble |
|---|---|---|---|---|
| clean | 0.9927 | 0.9927 | 0.9927 | 0.9936 |
| stain1 | 0.9763 | 0.9763 | 0.9754 | 0.9752 |
| stain2 | 0.7524 | 0.7524 | 0.7539 | 0.6840 |
| stain3 | 0.4364 | 0.4364 | 0.4358 | 0.4072 |
| stain4 | 0.2607 | 0.2607 | 0.2605 | 0.2455 |
| stain5 | 0.2014 | 0.2014 | 0.2014 | 0.1926 |
| blur1 | 0.9860 | 0.9860 | 0.9860 | 0.9863 |
| blur2 | 0.9398 | 0.9398 | 0.9395 | 0.9418 |
| blur3 | 0.8503 | 0.8503 | 0.8503 | 0.8585 |
| blur4 | 0.6977 | 0.6977 | 0.6998 | 0.6524 |
| blur5 | 0.5989 | 0.5989 | 0.6004 | 0.5063 |
| noise1 | 0.9828 | 0.9828 | 0.9830 | 0.9906 |
| noise2 | 0.9237 | 0.9237 | 0.9231 | 0.9588 |
| noise3 | 0.6106 | 0.6106 | 0.6115 | 0.7729 |
| noise4 | 0.5288 | 0.5288 | 0.5285 | 0.5519 |
| noise5 | 0.4382 | 0.4382 | 0.4396 | 0.4665 |


## Under shift: Mean entropy (uncertainty)

| Condition | Softmax | Temperature scaling | MC dropout | Deep ensemble |
|---|---|---|---|---|
| clean | 0.0152 | 0.0201 | 0.0161 | 0.0305 |
| stain1 | 0.0459 | 0.0580 | 0.0488 | 0.1229 |
| stain2 | 0.1811 | 0.2126 | 0.1943 | 0.2937 |
| stain3 | 0.1677 | 0.1962 | 0.1814 | 0.2552 |
| stain4 | 0.0965 | 0.1143 | 0.1051 | 0.1414 |
| stain5 | 0.0415 | 0.0507 | 0.0460 | 0.0653 |
| blur1 | 0.0233 | 0.0298 | 0.0250 | 0.0521 |
| blur2 | 0.0739 | 0.0931 | 0.0788 | 0.1699 |
| blur3 | 0.2265 | 0.2823 | 0.2387 | 0.4734 |
| blur4 | 0.5004 | 0.6034 | 0.5175 | 0.8713 |
| blur5 | 0.8325 | 0.9642 | 0.8495 | 1.1624 |
| noise1 | 0.0394 | 0.0500 | 0.0421 | 0.0526 |
| noise2 | 0.1130 | 0.1383 | 0.1202 | 0.1784 |
| noise3 | 0.3283 | 0.3686 | 0.3447 | 0.5162 |
| noise4 | 0.3025 | 0.3504 | 0.3349 | 0.5811 |
| noise5 | 0.3572 | 0.4066 | 0.4050 | 0.5494 |

Accuracy is expected to fall as severity rises. The test is whether uncertainty rises with it, and whether calibration holds.

Results can be viewd in: 

![Reliability diagrams](figures/reliability_224px.png)
![Risk–coverage curves](figures/risk_coverage_224px.png)
![Accuracy and ECE under shift](figures/shift_224px.png)
![Grad-CAM with sanity check](figures/gradcam_224px.png)


## Data

[BloodMNIST](https://medmnist.com/), from the MedMNIST v2 benchmark, derived from the peripheral blood cell dataset of [Acevedo et al. (2020)](https://doi.org/10.1016/j.dib.2020.105474).

| | |
|---|---|
| Images | 17,092 microscopy images of single blood cells |
| Classes | basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil, platelet |
| Official split | 11,959 train / 1,712 validation / 3,421 test |
| Source | Hospital Clinic of Barcelona, CellaVision DM96 analyser |
| Resolutions | 28, 64, 128 and 224 px (MedMNIST+). The resolution behind each result is stated with it |
| Licence | CC BY 4.0 |

**The images are not committed to this repository.** `data/` is git-ignored, and the `medmnist` package downloads the dataset on first run.

The official splits are used unchanged. The validation split is used for early stopping and for fitting the temperature, and nothing else. The test split is touched once per reported number.

## Tech stack

`Python 3.12` · `PyTorch` · `torchvision` · `MedMNIST` · `scikit-learn` · `NumPy` · `Matplotlib` · `Gradio` · `uv`

## Quickstart

```bash
# 1. Clone
git clone https://github.com/ayindemalik/Trustworthy-Medical-Image-Classification.git
cd Trustworthy-Medical-Image-Classification

# 2. Install (uv handles the virtual environment and the lockfile)
uv sync

# 3. Look at the data: class counts, example cells, simulated shift
uv run trustmed explore --size 224

# 4. Train (repeat with seeds 1-4 for the deep ensemble; a GPU is strongly recommended)
uv run trustmed train --size 224 --seed 0

# 5. Run every model on the clean and shifted test sets, and cache the outputs
uv run trustmed predict --size 224

# 6. Metrics, tables and charts, computed from the cached outputs in seconds
uv run trustmed evaluate --size 224

# 7. Grad-CAM grids and the sanity check
uv run trustmed explain --size 224
```

Add `--size 64 --limit 512` to any command for a quick test on a CPU. Output files from a quick test are named `*_limit512*` and never mix with real results.

The results in this README are produced on a free Colab T4 GPU, running the same commands as `python -m trustmed ...`.

## Project structure

```
Trustworthy-Medical-Image-Classification/
├── src/trustmed/
│   ├── cli.py            # command-line entry point
│   ├── config.py         # every tunable value, in one place
│   ├── data.py           # BloodMNIST loading, augmentation, simulated shift
│   ├── explore.py        # class counts and example pictures
│   ├── model.py          # ResNet-18 with a dropout head; saving, loading, running it
│   ├── train.py          # training loop, early stopping, checkpoints
│   ├── predict.py        # every model on every test condition, cached to outputs/
│   ├── calibration.py    # ECE, NLL, Brier, temperature scaling
│   ├── uncertainty.py    # entropy, risk–coverage, AURC, error-detection AUROC
│   ├── evaluate.py       # metrics, tables and charts from the cached outputs
│   ├── explain.py        # Grad-CAM and the weight-randomisation check
│   └── plots.py          # every chart, one shared style
├── tests/                # pytest: the metrics on cases with known answers
├── scripts/              # CPU vs GPU speed test
├── results/              # metrics JSON and markdown tables (committed)
├── figures/              # reliability diagrams, risk–coverage curves, Grad-CAM grids
├── data/                 # git-ignored; downloaded on first run
├── models/               # git-ignored checkpoints
├── outputs/              # git-ignored cached predictions
├── pyproject.toml
└── README.md
```

## How it works

```mermaid
flowchart LR
    subgraph Train["Training (5 seeds)"]
        A[BloodMNIST<br/>train split] --> B[ResNet-18<br/>fine-tune]
        B --> C[(Checkpoints)]
    end

    subgraph Calibrate["Calibration"]
        V[Validation split] --> T[Fit temperature T]
    end

    subgraph Evaluate["Evaluation"]
        X[Test split<br/>clean + shifted] --> P[Predict<br/>softmax · MC dropout · ensemble]
        C --> P
        T --> P
        P --> M[Accuracy · ECE · NLL]
        P --> S[Uncertainty →<br/>selective prediction]
        P --> G[Grad-CAM<br/>+ sanity check]
    end

    C --> T
```

Three details do most of the work.

- **Temperature scaling changes confidence, never the answer.** It divides every logit by one learned number, T. The ranking of classes is unchanged, so accuracy stays the same and only the confidence moves. T is fitted on the validation split. Fitting it on the test split would make the calibration result circular.
- **Deep ensembles are the strong baseline, MC dropout is the cheap one.** Five independently trained networks tend to disagree exactly where the data is ambiguous, and they have held up best under distribution shift in published comparisons (Ovadia et al., 2019). They cost five times the training. MC dropout costs nothing extra to train, and because the dropout sits only before the last layer, its 30 prediction passes re-run that one layer and nothing else. The comparison table is there to show whether the cheaper method is good enough.
- **A heatmap is only evidence if it depends on the model.** Some saliency methods produce nearly the same map from a randomly initialised network (Adebayo et al., 2018), which means they show image edges, not the model's reasoning. Grad-CAM is re-run after the trained weights are randomised, and the similarity between the two maps is reported. A trustworthy explanation should change.

## Roadmap

- [ ] Publish the results tables, reliability diagrams and risk–coverage curves
- [ ] Gradio demo: upload a cell image, get a prediction, a confidence, a "refer to expert" flag and a Grad-CAM overlay
- [ ] Real shift instead of simulated shift: test on blood-cell images from a different lab and scanner (for example Raabin-WBC, on the five white-cell classes both datasets share)
- [ ] Conformal prediction: prediction sets with a guaranteed coverage level, as an alternative to a single-label answer
- [ ] Published container image, then a hosted demo

## Limitations

- **Only normal cells.** BloodMNIST comes from people without haematological disease, so it contains no leukaemic or otherwise abnormal cells. How the model's uncertainty behaves on an abnormal cell is not measured here, and it is the case that matters most clinically.
- **One hospital, one analyser.** Simulated staining and blur are a stand-in for lab-to-lab variation, not a replacement for it.
- **Low resolution.** The images are downsampled from 360 px. Fine nuclear detail is lost, and Grad-CAM maps are coarse at the resolutions a CPU can train on.
- **Grad-CAM shows where, not why.** A heatmap on the nucleus says the model used the nucleus. It does not say what about the nucleus it used.
- **The ensemble members share a starting point.** All five start from the same ImageNet weights and differ only in the new head, the data order and the augmentation, so they are less diverse than networks trained from scratch.
- **Research code, not a medical device.** It is not validated for clinical use.

## Data citation

If you use this repository, please also cite the datasets it is built on:

- J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, B. Ni. *MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 10, 41 (2023).
- A. Acevedo, A. Merino, S. Alférez, Á. Molina, L. Boldú, J. Rodellar. *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems.* Data in Brief, 30, 105474 (2020).

## License

Released under the [MIT License](LICENSE). The MIT licence covers this source code only. BloodMNIST remains under CC BY 4.0.

## Author

**Dr. Maliki Moustapha** — PhD, Deep Learning & Computer Vision
📧 ayindemalik1@gmail.com · [LinkedIn](https://www.linkedin.com/in/maliki-moustapha-phd-525646169/) · [Portfolio](https://malikimoustapha.binotix.com/) · [Google Scholar](https://scholar.google.com/citations?user=bQe5fD0AAAAJ&hl=en) · [ORCID](https://orcid.org/0000-0001-9306-8554)
