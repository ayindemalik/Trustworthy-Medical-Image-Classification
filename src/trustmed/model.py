"""ResNet-18 with a dropout layer before the final classifier."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from trustmed import config
from trustmed.data import apply_shift, batch_indices, to_float

# The colour statistics of ImageNet, which the pretrained weights expect
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CellClassifier(nn.Module):
    """Input: images with values 0..1. Output: 8 class scores (logits)."""

    def __init__(self, pretrained: bool = True, dropout: float = config.DROPOUT):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_size = self.backbone.fc.in_features   # 512 numbers describe each image
        self.backbone.fc = nn.Identity()              # drop the 1000-class ImageNet layer
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_size, config.NUM_CLASSES)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def features(self, x):
        """Image -> 512 features. The expensive part."""
        return self.backbone((x - self.mean) / self.std)

    def classify(self, features):
        """512 features -> 8 logits. The cheap part."""
        return self.head(self.dropout(features))

    def forward(self, x):
        return self.classify(self.features(x))

    @property
    def last_conv_block(self):
        """Where Grad-CAM looks: the last block of convolutions."""
        return self.backbone.layer4


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def checkpoint_path(tag: str, seed: int):
    return config.MODELS_DIR / f"resnet18_{tag}_seed{seed}.pt"


def save(model: CellClassifier, path, **info):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **info}, path)


def load(path, device):
    """Rebuild the network and fill it with saved weights."""
    checkpoint = torch.load(path, map_location=device)
    model = CellClassifier(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def run_model(model, images, device, shift=None, severity=0, mc_passes=0, batch_size=256):
    """Logits [N, 8] for every image. With mc_passes > 0, also MC-dropout probabilities [N, 8]."""
    model.eval()
    logits, mc_probs = [], []
    for index in batch_indices(len(images), batch_size):
        x = to_float(images[index], device)
        if shift is not None:
            x = apply_shift(x, shift, severity, seed=int(index[0]))
        features = model.features(x)            # expensive: run once
        logits.append(model.classify(features).cpu())
        if mc_passes:
            model.dropout.train()               # dropout ON, for this one layer only
            passes = [model.classify(features).softmax(1) for _ in range(mc_passes)]
            model.dropout.eval()
            mc_probs.append(torch.stack(passes).mean(0).cpu())
    return torch.cat(logits), (torch.cat(mc_probs) if mc_passes else None)
