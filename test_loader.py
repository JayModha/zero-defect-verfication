from dataset_loader import load_images

X, y = load_images("mvtec_anomaly_detection/bottle")
print(f"✅ Loaded {len(X)} images")
print(f"Labels: {set(y)}")
