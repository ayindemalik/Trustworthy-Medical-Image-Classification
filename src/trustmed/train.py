"""Fine-tune one ResNet-18. Run it with five seeds to build the deep ensemble."""
import time

import torch
import torch.nn.functional as F

from trustmed import config
from trustmed.data import augment, batch_indices, load_split, to_float
from trustmed.model import CellClassifier, checkpoint_path, get_device, run_model, save


def train(size: int, seed: int, epochs: int = config.EPOCHS, limit: int | None = None):
    tag = config.run_tag(size, limit)
    device = get_device()
    torch.manual_seed(seed)   # the seed decides the new layer's starting weights, the batch order and the augmentation

    train_x, train_y = load_split("train", size, limit)
    val_x, val_y = load_split("val", size, limit)
    print(f"Training seed {seed} at {size}px on {device}: "
          f"{len(train_y)} training images, {len(val_y)} validation images")

    model = CellClassifier(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    final_path = checkpoint_path(tag, seed)
    partial_path = final_path.with_suffix(".partial.pt")   # renamed only when training finishes
    best_acc, epochs_without_gain = -1.0, 0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for index in batch_indices(len(train_y), config.BATCH_SIZE, shuffle=True, seed=seed * 1000 + epoch):
            if len(index) < config.BATCH_SIZE:
                continue   # skip the last, smaller batch: batch norm dislikes tiny batches
            x = augment(to_float(train_x[index], device))
            y = train_y[index].to(device)
            optimizer.zero_grad()                  # 1. clear old corrections
            loss = F.cross_entropy(model(x), y)    # 2. predict and measure the error
            loss.backward()                        # 3. work out the corrections
            optimizer.step()                       # 4. apply them
            loss_sum += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        val_logits, _ = run_model(model, val_x, device)
        val_acc = (val_logits.argmax(1) == val_y).float().mean().item()
        val_loss = F.cross_entropy(val_logits, val_y).item()
        improved = val_acc > best_acc
        print(f"epoch {epoch:2d} | train loss {loss_sum / seen:.4f} | val loss {val_loss:.4f} | "
              f"val acc {val_acc:.4f} | {time.perf_counter() - start:.0f}s" + ("  <- best so far, saved" if improved else ""))

        if improved:
            best_acc, epochs_without_gain = val_acc, 0
            save(model, partial_path, size=size, seed=seed, epoch=epoch, val_acc=val_acc)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.PATIENCE:
                print(f"No improvement for {config.PATIENCE} epochs, stopping early.")
                break

    partial_path.replace(final_path)
    print(f"Done. Best validation accuracy {best_acc:.4f}. Weights saved to {final_path}")
