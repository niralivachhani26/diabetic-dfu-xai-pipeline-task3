import os
import shutil
import pandas as pd

# 1. PASTE YOUR EXTRACTED KAGGLE FOLDER PATH HERE
kaggle_folder_path = r"F:\PHD\archive\DFU\Patches"

output_images_dir = "data/images"
output_csv_path = "data/dfuc_metadata.csv"
os.makedirs(output_images_dir, exist_ok=True)

records = []

subfolders = sorted([
    folder for folder in os.listdir(kaggle_folder_path)
    if os.path.isdir(os.path.join(kaggle_folder_path, folder))
])

print(f"Found folders: {subfolders}")

for label_number, folder_name in enumerate(subfolders):
    folder_path = os.path.join(kaggle_folder_path, folder_name)
    print(f"Copying '{folder_name}' as Label {label_number}...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            new_filename = f"class{label_number}_{filename}"
            shutil.copy2(
                os.path.join(folder_path, filename),
                os.path.join(output_images_dir, new_filename)
            )
            records.append({
                "image_id": new_filename,
                "label": label_number
            })

df = pd.DataFrame(records)
df.to_csv(output_csv_path, index=False)
print(f"\nSuccess! Copied {len(df)} images and created '{output_csv_path}'.")