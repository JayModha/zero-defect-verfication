import os
import cv2
import numpy as np

def load_images(path, target_size=(224, 224)):
    images = []
    labels = []
    for label_type in ['train/good', 'test/good', 'test/broken_small', 'test/broken_large']:
        full_path = os.path.join(path, label_type)
        label = 0 if 'good' in label_type else 1
        if os.path.exists(full_path):
            for img_file in os.listdir(full_path):
                img_path = os.path.join(full_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)
