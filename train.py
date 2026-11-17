import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader

from src.dataset import DFUDataset, get_dfu_transforms
from src.model import DFUClassifier, FocalLoss


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    use_amp = device.type == "cuda"
    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        if batch_idx % 5 == 0 or batch_idx == total_batches:
            print(f"  [Epoch {epoch:02d}/{total_epochs:02d}] Batch {batch_idx:02d}/{total_batches:02d} | Loss: {loss.item():.4f}", end="\r")

    print()  # New line after epoch finishes
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class="ovr")
    except ValueError:
        auc = 0.0

    return running_loss / len(loader.dataset), macro_f1, auc


def main():
    csv_path = "data/dfuc_metadata.csv"
    img_dir = "data/images"
    os.makedirs("assets", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device.type.upper()}")

    epochs = 10 if device.type == "cpu" else 25
    batch_size = 32
    lr = 3e-4

    df = pd.read_csv(csv_path)
    num_classes = int(df["label"].nunique())
    print(f"Detected {num_classes} classes in {csv_path} ({len(df)} total images).")

    if "patient_id" in df.columns:
        splitter = StratifiedGroupKFold(n_splits=5)
        train_idx, val_idx = next(splitter.split(df, df["label"], groups=df["patient_id"]))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(splitter.split(df, df["label"]))

    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_tf, val_tf = get_dfu_transforms(img_size=224)
    train_loader = DataLoader(DFUDataset(train_df, img_dir, train_tf), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(DFUDataset(val_df, img_dir, val_tf), batch_size=batch_size, shuffle=False, num_workers=0)

    class_counts = train_df["label"].value_counts().sort_index().values
    alpha_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float).to(device)
    alpha_weights = alpha_weights / alpha_weights.sum()

    model = DFUClassifier(num_classes=num_classes).to(device)

    # Speed up CPU training while keeping the last convolutional block trainable for Grad-CAM
    if device.type == "cpu":
        for param in model.backbone.features[:-1].parameters():
            param.requires_grad = False
        print("CPU detected: Froze early backbone layers (kept final feature block + classifier trainable).")

    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, epochs)
        val_loss, val_f1, val_auc = evaluate(model, val_loader, criterion, num_classes, device)
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "assets/best_dfu_model.pth")
            print("  --> Saved new best model checkpoint.")


if __name__ == "__main__":
    main()