import os
import cv2
import numpy as np

def load_custom_dataset(base_path, target_size=(224, 224)):
    images = []
    labels = []

    for folder_name, label in [("non_defective", 0), ("defective", 1)]:
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)
                labels.append(label)
            else:
                print(f"⚠️ Failed to read image: {img_path}")

    return np.array(images), np.array(labels)
