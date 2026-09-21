"""Grad-CAM: which part of the image drove the prediction, plus a check that the answer is real."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from trustmed import config, plots
from trustmed.data import load_split, to_float
from trustmed.model import CellClassifier, checkpoint_path, get_device, load, run_model


def grad_cam(model, x, class_index=None):
    """Heatmaps [N, H, W] between 0 and 1, for the given classes (default: the predicted ones)."""
    saved = {}

    def keep(module, inputs, output):
        saved["activations"] = output                                     # what the last block saw
        output.register_hook(lambda grad: saved.update(gradients=grad))   # how the class score reacts to it

    handle = model.last_conv_block.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            logits = model(x)
            if class_index is None:
                class_index = logits.argmax(dim=1)
            model.zero_grad()
            logits.gather(1, class_index[:, None]).sum().backward()
    finally:
        handle.remove()

    weights = saved["gradients"].mean(dim=(2, 3), keepdim=True)                # importance of each of the 512 channels
    cam = F.relu((weights * saved["activations"]).sum(dim=1, keepdim=True))    # weighted sum, positive evidence only
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)                    # scale each map to 0..1
    return cam.detach(), class_index


def explain(size: int, seed: int = 0, limit: int | None = None, sanity_images: int = 200, batch_size: int = 32):
    tag = config.run_tag(size, limit)
    device = get_device()
    model, _ = load(checkpoint_path(tag, seed), device)
    torch.manual_seed(seed)
    random_model = CellClassifier(pretrained=False).to(device).eval()   # same network, random weights

    images, labels = load_split("test", size, limit)
    logits, _ = run_model(model, images, device)
    confidence, predicted = logits.softmax(dim=1).max(dim=1)
    names = config.CLASS_NAMES

    # 1. For each class, the most confident correct example: what does the model look at?
    typical = []
    for c in range(config.NUM_CLASSES):
        right = ((labels == c) & (predicted == c)).nonzero()[:, 0]
        if len(right):
            typical.append(int(right[confidence[right].argmax()]))
    # 2. The eight least confident cases: what confuses it?
    unsure = confidence.argsort()[:8].tolist()

    figures = [
        ("gradcam", typical, [f"{names[labels[i]]}\n{confidence[i]:.2f}" for i in typical]),
        ("gradcam_uncertain", unsure,
         [f"true: {names[labels[i]]}\npred: {names[predicted[i]]}\n{confidence[i]:.2f}" for i in unsure]),
    ]
    for name, chosen, titles in figures:
        x = to_float(images[chosen], device)
        cams, classes = grad_cam(model, x)
        cams_random, _ = grad_cam(random_model, x, classes)
        plots.gradcam(x.cpu(), cams.cpu(), cams_random.cpu(), titles, f"{name}_{tag}.png")

    # 3. Sanity check: if the map barely changes when the weights are random, it is not explaining the model
    sample = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))[:sanity_images]
    correlations = []
    for chunk in sample.split(batch_size):
        x = to_float(images[chunk], device)
        cams, classes = grad_cam(model, x)
        cams_random, _ = grad_cam(random_model, x, classes)
        for trained, random in zip(cams.cpu().numpy(), cams_random.cpu().numpy()):
            if trained.std() > 0 and random.std() > 0:   # a flat map has no ranking to compare
                correlations.append(spearmanr(trained.ravel(), random.ravel()).statistic)

    summary = {
        "tag": tag, "seed": seed, "images_checked": len(sample), "images_compared": len(correlations),
        "spearman_mean": float(np.mean(correlations)), "spearman_std": float(np.std(correlations)),
        "meaning": "Rank correlation between Grad-CAM maps of the trained and the randomly initialised network. "
                   "Near 0 means the maps depend on what the model learned; near 1 means they do not.",
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"gradcam_sanity_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"Sanity check on {len(correlations)} images: trained vs random-weight maps, "
          f"Spearman {summary['spearman_mean']:.3f} +/- {summary['spearman_std']:.3f}")
