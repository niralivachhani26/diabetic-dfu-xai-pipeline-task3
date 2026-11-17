import os
import sys
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import DFUDataset, get_dfu_transforms
from src.explainability import GradCAM, overlay_cam_on_image
from src.model import DFUClassifier, FocalLoss


def test_end_to_end_dfu_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create synthetic sample image & metadata
        sample_img = np.full((256, 256, 3), [200, 160, 130], dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, "foot_01.jpg"), sample_img)

        df = pd.DataFrame([{"image_id": "foot_01", "label": 1, "patient_id": "P001"}])
        _, val_tf = get_dfu_transforms(img_size=256)
        dataset = DFUDataset(df=df, img_dir=tmp_dir, transform=val_tf)

        img_tensor, label = dataset[0]
        assert img_tensor.shape == (3, 256, 256)
        assert label.item() == 1

        # 2. Forward pass & Focal Loss check
        model = DFUClassifier(num_classes=4)
        model.eval()
        batch_x = img_tensor.unsqueeze(0)
        batch_x.requires_grad = True

        logits = model(batch_x)
        assert logits.shape == (1, 4)

        criterion = FocalLoss(gamma=2.0)
        loss = criterion(logits, label.unsqueeze(0))
        assert loss.item() > 0

        # 3. Grad-CAM verification on EfficientNet-V2-S last feature block
        cam = GradCAM(model=model, target_layer=model.backbone.features[-1])
        heatmap = cam.generate_cam(batch_x, target_class=1)
        cam.remove_hooks()

        assert heatmap.shape == (256, 256)
        assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0

        overlay = overlay_cam_on_image(sample_img, heatmap)
        assert overlay.shape == (256, 256, 3)

    print("End-to-end DFU pipeline & Grad-CAM test passed!")


if __name__ == "__main__":
    test_end_to_end_dfu_pipeline()