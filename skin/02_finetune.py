"""Fine-tune EfficientNet-B0 (ImageNet weights) on the HAM10000 subset.

Two-phase transfer learning, the standard recipe:
  Phase 1: freeze the backbone, train only the new 7-class head (fast,
           stops the random head from wrecking pretrained features).
  Phase 2: unfreeze everything, train at a low learning rate.

Runs on Apple Silicon GPU via the MPS backend. Saves the best checkpoint
(by validation balanced accuracy) plus a confusion matrix and training
curves to skin/results/.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

SKIN = Path(__file__).resolve().parent
DATA_DIR = SKIN / "data"
OUT_DIR = SKIN / "results"

IMG_SIZE = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 3
FT_EPOCHS = 8
HEAD_LR = 1e-3
FT_LR = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_loaders() -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # lesions have no canonical orientation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)
    assert train_ds.classes == val_ds.classes
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True
    )
    return train_loader, val_loader, train_ds.classes


def class_weights(train_ds: datasets.ImageFolder) -> torch.Tensor:
    counts = np.bincount([y for _, y in train_ds.samples])
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(y.numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return preds, targets, balanced_accuracy_score(targets, preds)


def train_epochs(model, loader, val_loader, criterion, optimizer, epochs, device, tag, history):
    best = -1.0
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            n += len(y)
        _, _, bal_acc = evaluate(model, val_loader, device)
        history.append({"phase": tag, "epoch": epoch, "loss": running / n, "val_bal_acc": bal_acc})
        print(
            f"[{tag}] epoch {epoch + 1}/{epochs}  loss {running / n:.4f}  "
            f"val bal-acc {bal_acc:.3f}  ({time.time() - t0:.0f}s)"
        )
        if bal_acc > best:
            best = bal_acc
            torch.save(model.state_dict(), OUT_DIR / "effnet_b0_best.pt")
    return best


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, classes = make_loaders()
    print(f"Classes: {classes}")
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_loader.dataset).to(device))
    history: list[dict] = []

    # Phase 1: head only
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=HEAD_LR)
    train_epochs(model, train_loader, val_loader, criterion, opt, HEAD_EPOCHS, device, "head", history)

    # Phase 2: full fine-tune
    for p in model.features.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    best = train_epochs(model, train_loader, val_loader, criterion, opt, FT_EPOCHS, device, "ft", history)
    print(f"Best val balanced accuracy: {best:.3f}")

    # Final eval with best checkpoint
    model.load_state_dict(torch.load(OUT_DIR / "effnet_b0_best.pt", weights_only=True))
    preds, targets, bal_acc = evaluate(model, val_loader, device)

    cm = confusion_matrix(targets, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (val, row-normalized) — bal-acc {bal_acc:.3f}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    with open(OUT_DIR / "training_log.json", "w") as f:
        json.dump({"classes": classes, "best_val_bal_acc": best, "history": history}, f, indent=2)
    with open(OUT_DIR / "classes.json", "w") as f:
        json.dump(classes, f)
    print(f"Saved checkpoint, confusion matrix and logs to {OUT_DIR}")


if __name__ == "__main__":
    main()
