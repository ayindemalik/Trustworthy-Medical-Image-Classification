"""Speed test: how long does one training epoch take on this machine?

Uses random images, so nothing needs downloading. The network does the
same amount of work whether the pixels are real or random.
"""
import math
import time

import torch
import torch.nn as nn
from torchvision.models import resnet18

TRAIN_IMAGES = 11_959   # size of the BloodMNIST training split
BATCH_SIZE = 32         # images processed together in one step
TIMED_STEPS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | CPU threads used by PyTorch: {torch.get_num_threads()}")

model = resnet18(num_classes=8).to(device)   # 8 blood-cell classes
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
model.train()

for size in (28, 64, 128, 224):
    images = torch.randn(BATCH_SIZE, 3, size, size, device=device)   # fake images
    labels = torch.randint(0, 8, (BATCH_SIZE,), device=device)       # fake labels

    def one_step():
        optimizer.zero_grad()                  # clear old gradients
        loss = loss_fn(model(images), labels)  # forward pass: predict, measure error
        loss.backward()                        # backward pass: work out the corrections
        optimizer.step()                       # apply the corrections to the weights

    one_step()  # warm-up: the first step is always slower, so we don't time it
    if device == "cuda":
        torch.cuda.synchronize()   # wait for the GPU to really finish before timing
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds_per_step = (time.perf_counter() - start) / TIMED_STEPS

    steps_per_epoch = math.ceil(TRAIN_IMAGES / BATCH_SIZE)
    minutes_per_epoch = seconds_per_step * steps_per_epoch / 60
    heatmap = math.ceil(size / 32)   # ResNet-18 shrinks the image 32x before the last layer
    print(f"{size:>3} px: {seconds_per_step:.2f} s/step -> about {minutes_per_epoch:.1f} min per epoch"
          f" | Grad-CAM grid {heatmap}x{heatmap}")
