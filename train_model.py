import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc, confusion_matrix
from collections import Counter
import json
import logging

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import backend as K


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):

        if y_pred.shape[-1] == 1: 
            y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        else:  
            y_pred_binary = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
            

        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        else:
            y_true = tf.cast(y_true, tf.float32)
        
        self.precision.update_state(y_true, y_pred_binary, sample_weight)
        self.recall.update_state(y_true, y_pred_binary, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def load_and_split_data_mvtec(category, target_size=(224, 224), defect_train_split=0.7):
    """
    Loads and splits data from the MVTec AD dataset with proper validation.
    Fixed to handle the correct MVTec structure:
    - train/good: non-defective images (label=0)
    - test/[various_folders]: all defective images (label=1)
    """
    base_dir = os.path.join("mvtec_anomaly_detection", category)
    train_good_dir = os.path.join(base_dir, "train", "good")
    test_root_dir = os.path.join(base_dir, "test")

    if not os.path.exists(base_dir):
        logging.error(f"Dataset directory not found: {base_dir}")
        return (None,)*4
    
    if not os.path.exists(train_good_dir):
        logging.error(f"Training good directory not found: {train_good_dir}")
        return (None,)*4
    
    if not os.path.exists(test_root_dir):
        logging.error(f"Test directory not found: {test_root_dir}")
        return (None,)*4

    X_train, y_train = [], []
    X_test, y_test = [], []

    def load_images_from_dir(directory, label, image_list, label_list):
        if not os.path.exists(directory):
            logging.warning(f"Directory does not exist: {directory}")
            return
            
        logging.info(f"Loading images from: {directory} with label: {label}")
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for img_file in files:
            path = os.path.join(directory, img_file)
            try:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    if np.sum(img) > 0:
                        image_list.append(img)
                        label_list.append(label)
                else:
                    logging.warning(f"Could not load image file: {path}")
            except Exception as e:
                logging.error(f"Error processing image {path}: {e}")

    # Load non-defective training images from train/good
    load_images_from_dir(train_good_dir, 0, X_train, y_train)
    logging.info(f"Loaded {len([y for y in y_train if y == 0])} non-defective training images")

    # Load ALL defective images from test folder (all subfolders contain defective images)
    all_defect_images, all_defect_labels = [], []
    logging.info(f"Loading defective images from: {test_root_dir}")
    
    for subfolder in os.listdir(test_root_dir):
        defect_type_dir = os.path.join(test_root_dir, subfolder)
        if os.path.isdir(defect_type_dir):
            logging.info(f"Loading defective images from subfolder: {subfolder}")
            load_images_from_dir(defect_type_dir, 1, all_defect_images, all_defect_labels)
    
    if not all_defect_images:
        logging.error("No defective images found in test directory. Cannot proceed with binary classification.")
        return (None,)*4

    logging.info(f"Total defective images loaded: {len(all_defect_images)}")

    # Split defective images between train and test
    X_defect_train, X_defect_test, y_defect_train, y_defect_test = train_test_split(
        all_defect_images, all_defect_labels, 
        train_size=defect_train_split, 
        random_state=SEED, 
        shuffle=True
    )
    
    logging.info(f"Defective images for training: {len(X_defect_train)}")
    logging.info(f"Defective images for testing: {len(X_defect_test)}")

    # Add defective images to train set
    X_train.extend(X_defect_train)
    y_train.extend(y_defect_train)
    
    # Add defective images to test set
    X_test.extend(X_defect_test)
    y_test.extend(y_defect_test)

    # We need some non-defective images in test set too for proper evaluation
    # Split some non-defective images from training set
    if len([y for y in y_train if y == 0]) > 20:  # If we have enough non-defective images
        non_defect_indices = [i for i, y in enumerate(y_train) if y == 0]
        
        # Take 20% of non-defective images for test set
        test_size = max(10, int(len(non_defect_indices) * 0.2))
        np.random.seed(SEED)
        test_indices = np.random.choice(non_defect_indices, size=test_size, replace=False)
        
        # Move selected non-defective images to test set
        X_train_array = np.array(X_train)
        y_train_array = np.array(y_train)
        
        X_test_non_defect = X_train_array[test_indices]
        y_test_non_defect = y_train_array[test_indices]
        
        # Remove from train set
        remaining_indices = [i for i in range(len(X_train)) if i not in test_indices]
        X_train = [X_train[i] for i in remaining_indices]
        y_train = [y_train[i] for i in remaining_indices]
        
        # Add to test set
        X_test.extend(X_test_non_defect.tolist())
        y_test.extend(y_test_non_defect.tolist())
        
        logging.info(f"Moved {len(test_indices)} non-defective images to test set")

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Validate we have both classes in both sets
    if len(np.unique(y_train)) < 2:
        logging.error("Training set doesn't contain both classes!")
        return (None,)*4
    
    if len(np.unique(y_test)) < 2:
        logging.error("Test set doesn't contain both classes!")
        return (None,)*4

    logging.info(f"Final training set: {len(X_train)} images -> {Counter(y_train)}")
    logging.info(f"Final test set: {len(X_test)} images -> {Counter(y_test)}")

    return X_train, y_train, X_test, y_test


def build_efficientnet_model(input_shape=(224, 224, 3)):
    """
    Builds a more robust transfer learning model using EfficientNetB0.
    """
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False 


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x) 
    x = Dropout(0.3)(x)  
    x = Dense(128, activation="relu")(x)  
    x = Dropout(0.2)(x) 
    output = Dense(1, activation="sigmoid", name="predictions")(x)  

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def preprocess_data(X, y):
    """
    Preprocess the data with proper normalization and validation.
    """

    X = X.astype("float32") / 255.0
    

    y = y.astype(np.int32)
    

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logging.warning("Found NaN or Inf values in data, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y


def calculate_class_weights(y):
    """Calculate balanced class weights"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    logging.info(f"Calculated class weights: {class_weight_dict}")
    return class_weight_dict


def train_with_better_strategy(model, base_model, X_train, y_train, X_val, y_val, 
                             args, save_dir, class_weight):
    """
    Improved training strategy with better monitoring and early stopping.
    """
    best_model_path = os.path.join(save_dir, "best_anomaly_model.keras")
    csv_logger_path = os.path.join(save_dir, "training_log.csv")
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rotation_range=10,     
        zoom_range=0.05,        
        width_shift_range=0.05, 
        height_shift_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=[0.95, 1.05] 
    )
    

    logging.info("\n--- Phase 1: Training classification head (EfficientNetB0 frozen) ---")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate_phase1,
            clipnorm=1.0 
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", F1Score()]
    )

    callbacks_phase1 = [
        EarlyStopping(
            patience=7, 
            monitor='val_loss',
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3, 
            verbose=1,
            mode='min',
            min_lr=1e-7
        ),
        CSVLogger(csv_logger_path, append=False)
    ]


    steps_per_epoch = max(1, len(X_train) // args.batch_size)
    
    history1 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=args.batch_size, seed=SEED),
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=args.epochs_phase1,
        class_weight=class_weight,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    logging.info("Phase 1 training complete.")

    # --- Phase 2: Fine-tuning (if model is performing reasonably) ---
    # Check if we should proceed with fine-tuning
    best_val_loss = min(history1.history['val_loss'])
    if best_val_loss > 1.0:  # If loss is too high, skip fine-tuning
        logging.warning("Phase 1 validation loss too high, skipping fine-tuning")
        return history1, None
    
    logging.info("\n--- Phase 2: Fine-tuning model ---")
    

    base_model.trainable = True
    
   
    fine_tune_at = max(100, len(base_model.layers) - 50)  
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Much lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate_phase2,
            clipnorm=1.0
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", F1Score()]
    )

    callbacks_phase2 = [
        EarlyStopping(
            patience=8,  
            monitor='val_loss',
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4, 
            verbose=1,
            mode='min',
            min_lr=1e-8
        ),
        CSVLogger(csv_logger_path, append=True)
    ]

    history2 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=args.batch_size, seed=SEED),
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=args.epochs_phase2,
        class_weight=class_weight,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    logging.info("Phase 2 fine-tuning complete.")
    
    return history1, history2


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def plot_training_history(history_combined, save_dir):
    """Plots accuracy, loss, and F1-score from training history."""
    metrics_to_plot = {
        'Loss': ('loss', 'val_loss'),
        'Accuracy': ('accuracy', 'val_accuracy'),
        'F1-Score': ('f1_score_metric', 'val_f1_score_metric')
    }
    
    for title, (train_metric, val_metric) in metrics_to_plot.items():
        if train_metric not in history_combined.history:
            logging.warning(f"Metric {train_metric} not found in history, skipping plot")
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(history_combined.history[train_metric], label=f'Train {title}')
        plt.plot(history_combined.history[val_metric], label=f'Validation {title}')
        
        if 'phase_split_epoch' in history_combined.history:
            plt.axvline(x=history_combined.history['phase_split_epoch'], color='r', linestyle='--', label='Fine-Tuning Start')
            
        plt.title(f'Model {title} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_curve.png'))
        plt.close()
    
    logging.info("Training history plots saved.")

def plot_precision_recall(y_true, y_scores, save_dir):
    """Plots the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.close()
    logging.info(f"Precision-Recall curve (AUC={pr_auc:.2f}) saved.")

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plots and saves the Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Defective', 'Defective'], rotation=45)
    plt.yticks(tick_marks, ['Non-Defective', 'Defective'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    logging.info("Confusion matrix plot saved.")


def main():
    parser = argparse.ArgumentParser(description="Train an anomaly detection model on MVTec AD dataset.")
    parser.add_argument("--category", type=str, required=True, help="MVTec AD dataset category (e.g., 'bottle', 'capsule').")
    parser.add_argument("--epochs_phase1", type=int, default=12, help="Epochs for Phase 1 (frozen base model).")
    parser.add_argument("--epochs_phase2", type=int, default=15, help="Epochs for Phase 2 (fine-tuning).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Proportion of training data for validation.")
    parser.add_argument("--learning_rate_phase1", type=float, default=1e-3, help="Learning rate for Phase 1.")
    parser.add_argument("--learning_rate_phase2", type=float, default=1e-5, help="Learning rate for Phase 2.")
    parser.add_argument("--defect_train_split", type=float, default=0.7, help="Proportion of defective images for training.")
    args = parser.parse_args()

    # --- Load Data ---
    logging.info(f"Starting data loading for category: {args.category}")
    X_train_full, y_train_full, X_test, y_test = load_and_split_data_mvtec(
        args.category, defect_train_split=args.defect_train_split
    )
    
    if X_train_full is None:
        logging.error("Data loading failed. Exiting.")
        return


    X_train_full, y_train_full = preprocess_data(X_train_full, y_train_full)
    X_test, y_test = preprocess_data(X_test, y_test)

    logging.info(f"Total training samples: {len(X_train_full)} -> {Counter(y_train_full)}")
    logging.info(f"Total test samples: {len(X_test)} -> {Counter(y_test)}")

    train_counter = Counter(y_train_full)
    if len(train_counter) < 2:
        logging.error("Training data contains only one class. Cannot train binary classifier.")
        return
        
    imbalance_ratio = max(train_counter.values()) / min(train_counter.values())
    logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")


    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        stratify=y_train_full,
        test_size=args.validation_split,
        random_state=SEED
    )
    
    logging.info(f"Train samples: {len(X_train)} -> {Counter(y_train)}")
    logging.info(f"Validation samples: {len(X_val)} -> {Counter(y_val)}")

    class_weight = calculate_class_weights(y_train)


    model, base_model = build_efficientnet_model()
    

    logging.info("Model Architecture:")
    model.summary(line_length=120)


    save_dir = os.path.join("models", args.category)
    os.makedirs(save_dir, exist_ok=True)


    history1, history2 = train_with_better_strategy(
        model, base_model, X_train, y_train, X_val, y_val,
        args, save_dir, class_weight
    )


    logging.info("\n--- Final Evaluation on Test Set ---")
    

    best_model_path = os.path.join(save_dir, "best_anomaly_model.keras")
    try:
        model = tf.keras.models.load_model(
            best_model_path, 
            custom_objects={'F1Score': F1Score}
        )
        logging.info("Best model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading best model: {e}. Using current model.")


    y_pred_proba = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred_binary = (y_pred_proba > 0.5).astype(int).flatten()
    

    if y_test.ndim > 1:
        y_test = y_test.flatten()


    report_text = classification_report(y_test, y_pred_binary, digits=4)
    logging.info("Classification Report:\n" + report_text)
    

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)


    report_dict = classification_report(y_test, y_pred_binary, digits=4, output_dict=True)
    
    overall_metrics = {
        'category': args.category,
        'test_accuracy': float(report_dict['accuracy']),
        'test_f1_defective': float(report_dict.get('1', {}).get('f1-score', 0.0)),
        'test_precision_defective': float(report_dict.get('1', {}).get('precision', 0.0)),
        'test_recall_defective': float(report_dict.get('1', {}).get('recall', 0.0)),
        'test_f1_macro': float(report_dict.get('macro avg', {}).get('f1-score', 0.0)),
        'class_distribution_train': convert_numpy_types(dict(Counter(y_train))),
        'class_distribution_test': convert_numpy_types(dict(Counter(y_test)))
    }
    

    with open(os.path.join(save_dir, "metrics_summary.json"), "w") as f:
        json.dump(overall_metrics, f, indent=4)
    
    logging.info(f"Final Test Accuracy: {overall_metrics['test_accuracy']:.4f}")
    logging.info(f"Final Test F1 (Defective): {overall_metrics['test_f1_defective']:.4f}")


    if history2 is not None:

        combined_history = tf.keras.callbacks.History()
        combined_history.history = {}
        
        for key in history1.history.keys():
            combined_history.history[key] = history1.history[key] + history2.history[key]
        
        combined_history.history['phase_split_epoch'] = len(history1.history['loss']) - 1
    else:
        combined_history = history1

    plot_training_history(combined_history, save_dir)
    plot_precision_recall(y_test, y_pred_proba.flatten(), save_dir)
    plot_confusion_matrix(y_test, y_pred_binary, save_dir)

    logging.info(f"Training and evaluation complete for category: {args.category}")
    logging.info(f"All results saved in: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()