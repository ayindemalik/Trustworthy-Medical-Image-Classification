"""Load BloodMNIST, cut it into batches, augment it, and simulate a different lab."""
import math

import torch
import torchvision.transforms.v2.functional as TF
from medmnist import BloodMNIST

from trustmed import config


def load_split(split: str, size: int, limit: int | None = None):
    """Images as uint8 [N, H, W, 3] and labels as int64 [N] for one split.

    `limit` keeps a fixed random subset, for quick tests on a laptop.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    file = config.DATA_DIR / f"bloodmnist_{size}.npz"
    dataset = BloodMNIST(split=split, size=size, root=str(config.DATA_DIR),
                         download=not file.exists())
    images = torch.from_numpy(dataset.imgs)
    labels = torch.from_numpy(dataset.labels[:, 0]).long()
    if limit is not None and limit < len(labels):
        keep = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))[:limit]
        images, labels = images[keep], labels[keep]
    return images, labels


def batch_indices(n: int, batch_size: int, shuffle: bool = False, seed: int = 0):
    """Split 0..n-1 into batches of positions, shuffled if asked."""
    if shuffle:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    else:
        order = torch.arange(n)
    return order.split(batch_size)


def to_float(images: torch.Tensor, device) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float [N, 3, H, W] between 0 and 1, on the device."""
    return images.to(device).permute(0, 3, 1, 2).float().div(255)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random mirror and quarter turns. A cell has no 'up', so each image counts as eight."""
    mirror = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(mirror[:, None, None, None], x.flip(3), x)
    turns = torch.randint(0, 4, (len(x),), device=x.device)
    for k in (1, 2, 3):
        chosen = turns == k
        if chosen.any():
            x[chosen] = torch.rot90(x[chosen], k, dims=(2, 3))
    return x


def apply_shift(x: torch.Tensor, kind: str, severity: int, seed: int = 0) -> torch.Tensor:
    """Pretend the image came from another lab. Severity 1 is mild, 5 is strong."""
    if kind == "stain":      # different stain batch: colours drift
        x = TF.adjust_hue(x, 0.02 * severity)
        x = TF.adjust_saturation(x, 1 + 0.15 * severity)
    elif kind == "blur":     # microscope slightly out of focus
        sigma = severity * x.shape[-1] / 200
        kernel = 2 * math.ceil(3 * sigma) + 1
        x = TF.gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    elif kind == "noise":    # noisier camera sensor
        noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))
        x = x + 0.025 * severity * noise.to(x.device)
    else:
        raise ValueError(f"Unknown shift: {kind}")
    return x.clamp(0, 1)
