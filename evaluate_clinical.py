import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.dataset import DFUDataset, get_dfu_transforms
from src.model import DFUClassifier


@torch.no_grad()
def run_clinical_audit():
    csv_path = "data/dfuc_metadata.csv"
    img_dir = "data/images"
    weights_path = "assets/best_dfu_model.pth"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at '{weights_path}'. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path)

    # Recreate exact validation split
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _, val_idx = next(splitter.split(df, df["label"]))
    val_df = df.iloc[val_idx]

    _, val_tf = get_dfu_transforms(img_size=224)
    val_loader = DataLoader(DFUDataset(val_df, img_dir, val_tf), batch_size=32, shuffle=False, num_workers=0)

    model = DFUClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_targets, all_preds, all_ulcer_probs = [], [], []

    for images, targets in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_targets.extend(targets.numpy())
        all_preds.extend(preds.cpu().numpy())
        # Label 0 = Abnormal (Ulcer), Label 1 = Normal (Healthy)
        all_ulcer_probs.extend(probs[:, 0].cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    ulcer_probs = np.array(all_ulcer_probs)

    # Confusion Matrix where Class 0 = Positive (Ulcer) and Class 1 = Negative (Normal)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    auc = roc_auc_score((y_true == 0).astype(int), ulcer_probs)

    print("\n" + "=" * 55)
    print("       CLINICAL VALIDATION REPORT (DFU DETECTION)")
    print("=" * 55)
    print(f"Total Validation Scans : {len(y_true)}")
    print(f"Sensitivity (Recall)   : {sensitivity * 100:.2f}%  (Ulcers caught)")
    print(f"Specificity            : {specificity * 100:.2f}%  (Healthy cleared)")
    print(f"Precision (PPV)        : {precision * 100:.2f}%")
    print(f"ROC-AUC Score          : {auc:.4f}")
    print("-" * 55)
    print("Confusion Matrix:")
    print(f"                 Predicted Ulcer   Predicted Normal")
    print(f"Actual Ulcer     {tp:<17} {fn}")
    print(f"Actual Normal    {fp:<17} {tn}")
    print("=" * 55)


if __name__ == "__main__":
    run_clinical_audit()