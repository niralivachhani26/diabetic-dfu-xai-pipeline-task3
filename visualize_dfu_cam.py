import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataset import get_dfu_transforms
from src.explainability import GradCAM, overlay_cam_on_image
from src.model import DFUClassifier

CLASS_NAMES = {
    0: "Abnormal (Diabetic Foot Ulcer)",
    1: "Normal (Healthy Skin)"
}


def generate_dfu_explanation(
    image_path: str = None,
    weights_path: str = "assets/best_dfu_model.pth",
    output_path: str = "assets/dfu_gradcam_report.png",
    num_classes: int = 2,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Auto-pick a real ulcer image from data/dfuc_metadata.csv if no path is passed
    if image_path is None and os.path.exists("data/dfuc_metadata.csv"):
        df = pd.read_csv("data/dfuc_metadata.csv")
        ulcer_rows = df[df["label"] == 0]
        sample_row = ulcer_rows.iloc[0] if not ulcer_rows.empty else df.iloc[0]
        image_path = os.path.join("data/images", str(sample_row["image_id"]))

    # 2. Load image (or synthesize fallback if data folder is missing)
    if image_path and os.path.exists(image_path):
        print(f"Generating Grad-CAM explanation for: {image_path}")
        bgr = cv2.imread(image_path)
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_rgb = cv2.resize(raw_rgb, (224, 224))
    else:
        print("No dataset image found. Synthesizing plantar lesion sample...")
        raw_rgb = np.full((224, 224, 3), [215, 175, 145], dtype=np.uint8)
        cv2.circle(raw_rgb, (112, 112), 28, (195, 75, 70), -1)
        cv2.circle(raw_rgb, (112, 112), 12, (130, 30, 35), -1)

    # 3. Preprocess tensor (matches 224x224 training size)
    _, val_tf = get_dfu_transforms(img_size=224)
    input_tensor = val_tf(Image.fromarray(raw_rgb)).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # 4. Load model & trained weights
    model = DFUClassifier(num_classes=num_classes).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded trained checkpoint: {weights_path}")
    else:
        print("Warning: Trained checkpoint not found. Running with ImageNet weights.")
    model.eval()

    # 5. Predict class & confidence
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

    # 6. Extract Grad-CAM from final convolutional block of EfficientNet-V2-S
    target_layer = model.backbone.features[-1]
    cam_engine = GradCAM(model=model, target_layer=target_layer)
    heatmap = cam_engine.generate_cam(input_tensor, target_class=pred_idx)
    cam_engine.remove_hooks()

    overlay_rgb = overlay_cam_on_image(raw_rgb, heatmap, alpha=0.45)

    # 7. Render 3-panel clinical report figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("1. Clinical Plantar Scan", raw_rgb, None),
        ("2. Grad-CAM Attention Map", heatmap, "jet"),
        (f"3. {CLASS_NAMES.get(pred_idx, pred_idx)} ({confidence:.1f}%)", overlay_rgb, None),
    ]

    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved clinical Grad-CAM figure to: {output_path}")


if __name__ == "__main__":
    generate_dfu_explanation()