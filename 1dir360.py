"""
HAND GESTURE RECOGNITION - SIMPLE VERSION (GRU then KAN)
WITHOUT KAN VISUALIZATION (removed due to performance issues)
UNI-DIRECTIONAL GRU
MODIFIED: GRU_HIDDEN_SIZE = 128, KAN = [128,64,32,7]
✅ NO LINEAR LAYER - KAN outputs logits directly
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from kan import KAN  # المكتبة الحقيقية
import glob
import os
import warnings
import seaborn as sns
import time

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    TIME_POINTS = 360  # 4 seconds * 90 Hz
    FEATURES = 14
    CLASSES = 7
    SAMPLING_RATE = 90

    # KAN parameters
    KAN_GRID_SIZE = 5
    KAN_SPLINE_ORDER = 3

    # GRU parameters - UNI-DIRECTIONAL (MODIFIED: changed from 360 to 128)
    GRU_HIDDEN_SIZE = 128  # ✅ تم التعديل من 360 إلى 128
    GRU_NUM_LAYERS = 2

    # Training parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    EPOCHS = 100
    DROPOUT_RATE = 0.3

    DATA_FOLDER = "my_hand_data"
    MODEL_SAVE_PATH = "hand_gesture_gru_kan_unidirectional_1dir360.pth"


# ==================== SIMPLE MODEL: GRU then KAN ====================
class HandGestureGRUKAN(nn.Module):
    """Simple version without attention - UNI-DIRECTIONAL GRU
       ✅ NO LINEAR LAYER - KAN outputs logits directly (width ends with num_classes)
    """

    def __init__(self, input_features=14, num_classes=7):
        super().__init__()

        print("\n" + "=" * 60)
        print("🧠 BUILDING HAND GESTURE RECOGNITION MODEL (UNI-DIRECTIONAL)")
        print("✅ NO LINEAR LAYER - KAN outputs logits directly")
        print("=" * 60)

        # GRU for temporal patterns - UNI-DIRECTIONAL
        print(f"   • UNI-DIRECTIONAL GRU Layer: {input_features} → {Config.GRU_HIDDEN_SIZE}")
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=False,  # UNI-DIRECTIONAL ONLY
            dropout=Config.DROPOUT_RATE if Config.GRU_NUM_LAYERS > 1 else 0
        )

        self.gru_output_size = Config.GRU_HIDDEN_SIZE  # UNI-DIRECTIONAL: لا تضرب في 2
        print(f"   • GRU Output Size: {self.gru_output_size} (UNI-DIRECTIONAL)")

        # KAN for classification (MODIFIED: changed dimensions to match GRU_HIDDEN_SIZE=128)
        # ✅ العرض الأخير = num_classes, لذا لا حاجة لطبقة Linear منفصلة
        print(f"   • KAN Layer: {self.gru_output_size} → 64 → 32 → {num_classes}")
        self.kan = KAN(
            width=[self.gru_output_size, 64, 32, num_classes],  # ✅ KAN يخرج logits
            grid=Config.KAN_GRID_SIZE,
            k=Config.KAN_SPLINE_ORDER,
            seed=42
        )

        self.layer_norm = nn.LayerNorm(input_features)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        print("✅ Model created successfully!")
        print(f"   Architecture: Input → LayerNorm → UNI-DIRECTIONAL GRU → KAN → Output")
        print(f"   ✅ KAN outputs {num_classes} logits directly (NO Linear layer)")

    def forward(self, x):
        # Normalize
        x = self.layer_norm(x)

        # GRU (UNI-DIRECTIONAL)
        gru_out, _ = self.gru(x)  # [batch, time, hidden] - NO *2

        # Take last output (contains all sequence info)
        last_output = gru_out[:, -1, :]  # [batch, hidden]
        last_output = self.dropout(last_output)

        # KAN for final decision (outputs logits directly, no Linear)
        output = self.kan(last_output)  # [batch, num_classes]

        return output


# ==================== DATA PROCESSOR ====================
class HandGestureDataProcessor:
    def __init__(self, target_length=360):
        self.target_length = target_length
        self.required_features = [
            'palm_x', 'palm_y', 'palm_z',
            'palm_velocity_x', 'palm_velocity_y', 'palm_velocity_z',
            'grab_strength', 'pinch_strength', 'pinch_distance',
            'thumb_extended', 'index_extended', 'middle_extended',
            'ring_extended', 'pinky_extended'
        ]

        self.gesture_keywords = {
            'open': 'open',
            'close': 'close',
            'grab': 'grab',
            'push': 'push',
            'pull': 'pull',
            'raise': 'raise',
            'lower': 'lower'
        }

    def extract_gesture_name(self, filename: str) -> Optional[str]:
        filename_lower = filename.lower()
        for keyword, gesture in self.gesture_keywords.items():
            if keyword in filename_lower:
                return gesture
        return None

    def load_single_file(self, file_path: str) -> Tuple[np.ndarray, str]:
        try:
            df = pd.read_csv(file_path)
            gesture_name = self.extract_gesture_name(os.path.basename(file_path))

            if not gesture_name:
                return None, None

            # Ensure all features exist
            for feature in self.required_features:
                if feature not in df.columns:
                    df[feature] = 0.0

            feature_data = df[self.required_features].values.astype(np.float32)

            # Handle sequence length
            if len(feature_data) < self.target_length:
                padding = np.zeros((self.target_length - len(feature_data), len(self.required_features)))
                feature_data = np.vstack([feature_data, padding])
            else:
                feature_data = feature_data[:self.target_length]

            # Normalize
            normalized_data = self.normalize_features(feature_data)

            return normalized_data, gesture_name

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None, None

    def normalize_features(self, data: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(data)
        for i in range(data.shape[1]):
            feature = data[:, i]
            if np.std(feature) > 0.001:
                normalized[:, i] = (feature - np.mean(feature)) / np.std(feature)
            else:
                normalized[:, i] = feature - np.mean(feature)
        return normalized

    def load_all_files(self, data_folder: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print(f"📁 Loading data from: {data_folder}")

        csv_pattern = os.path.join(data_folder, "*.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_folder}")

        print(f"Found {len(csv_files)} CSV files")

        all_sequences = []
        all_labels = []
        file_paths = []

        for file_path in csv_files:
            sequence, label = self.load_single_file(file_path)
            if sequence is not None and label is not None:
                all_sequences.append(sequence)
                all_labels.append(label)
                file_paths.append(file_path)

        if not all_sequences:
            raise ValueError("No valid data loaded!")

        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_labels)

        print(f"\n✅ Successfully loaded data:")
        print(f"   Total sequences: {len(X)}")
        print(f"   Sequence shape: {X.shape}")

        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📊 Class distribution:")
        for gesture, count in zip(unique, counts):
            percentage = (count / len(y)) * 100
            print(f"   {gesture}: {count} samples ({percentage:.1f}%)")

        return X, y, file_paths


# ==================== DATASET CLASS ====================
class HandGestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(label_encoder.transform(y))
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== VISUALIZATION ====================
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
    axes[0].set_title('Training History - Loss (UNI-DIRECTIONAL GRU)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', alpha=0.7)
    axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', alpha=0.7)
    axes[1].set_title('Training History - Accuracy (UNI-DIRECTIONAL GRU)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_unidirectional.png', dpi=100, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names, num_samples=10):
    """Plot feature importance based on model activations"""
    print("\n📊 Analyzing feature importance...")

    try:
        model.eval()

        # Create random data for analysis
        device = next(model.parameters()).device
        dummy_input = torch.randn(num_samples, Config.TIME_POINTS, Config.FEATURES).to(device)

        with torch.no_grad():
            # Get GRU output
            dummy_input_norm = model.layer_norm(dummy_input)
            gru_out, _ = model.gru(dummy_input_norm)

            # Calculate mean absolute values
            feature_importance = torch.mean(torch.abs(gru_out), dim=(0, 1)).cpu().numpy()

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # If too many features, show top 20
            if len(feature_importance) > 20:
                indices = np.argsort(feature_importance)[-20:]  # Top 20 features
                sorted_importance = feature_importance[indices]
                # Create labels
                labels = [f'Feature {i}' for i in indices]
            else:
                sorted_importance = feature_importance
                labels = [f'Feature {i}' for i in range(len(feature_importance))]

            # Create horizontal bar chart
            y_pos = np.arange(len(sorted_importance))
            ax.barh(y_pos, sorted_importance, color='steelblue', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Average Absolute Activation')
            ax.set_title('Feature Importance (UNI-DIRECTIONAL GRU)')
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig('feature_importance_unidirectional.png', dpi=100, bbox_inches='tight')
            plt.show()
            print("✅ Feature importance plot created successfully!")

    except Exception as e:
        print(f"⚠️  Could not create feature importance plot: {e}")


# ==================== TRAINING FUNCTION ====================
def train_hand_gesture_model():
    """Main training function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("📁 LOADING HAND GESTURE DATA")
    print("=" * 60)

    processor = HandGestureDataProcessor(target_length=Config.TIME_POINTS)

    # Check if data folder exists
    if not os.path.exists(Config.DATA_FOLDER):
        print(f"❌ Data folder '{Config.DATA_FOLDER}' not found!")
        print(f"📁 Please create folder '{Config.DATA_FOLDER}' and add CSV files")
        return None, None

    X, y, file_paths = processor.load_all_files(Config.DATA_FOLDER)

    # Check if we have data
    if len(X) == 0:
        print(f"❌ No valid data found in '{Config.DATA_FOLDER}'")
        print("💡 Please add CSV files with the correct format")
        return None, None

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n🎯 Encoded gestures: {list(label_encoder.classes_)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} sequences")
    print(f"   Validation: {len(X_val)} sequences")
    print(f"   Test: {len(X_test)} sequences")

    # Create datasets and dataloaders
    train_dataset = HandGestureDataset(X_train, y_train, label_encoder)
    val_dataset = HandGestureDataset(X_val, y_val, label_encoder)
    test_dataset = HandGestureDataset(X_test, y_test, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # Create model
    print("\n" + "=" * 60)
    print("🧠 BUILDING MODEL (UNI-DIRECTIONAL GRU)")
    print("=" * 60)
    model = HandGestureGRUKAN(
        input_features=Config.FEATURES,
        num_classes=Config.CLASSES
    )

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Architecture: Input({Config.FEATURES}) → LayerNorm → UNI-DIRECTIONAL GRU({Config.GRU_HIDDEN_SIZE}) → KAN → Output({Config.CLASSES})")
    print(f"   GRU Configuration: {Config.GRU_NUM_LAYERS} layers, UNI-DIRECTIONAL")
    print(f"   ✅ NO Linear layer - KAN outputs logits directly")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0
    best_model_state = None
    patience_counter = 0

    print(f"\n🚀 Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)

        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'\n📈 Epoch [{epoch + 1:3d}/{Config.EPOCHS}]')
            print(f'   📉 Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%')
            print(f'   📊 Val Loss:   {avg_val_loss:.4f}, Acc:   {val_accuracy:.2f}%')
            print(f'   🏆 Best Val Acc: {best_val_accuracy:.2f}%')
            print(f'   📚 Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if patience_counter >= 15:
            print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test with time measurement
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL")
    print("=" * 60)

    model.eval()
    test_correct, test_total = 0, 0
    all_predictions, all_true_labels = [], []

    # Start timing
    start_time = time.time()

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)

            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())

    # End timing
    end_time = time.time()
    test_duration = end_time - start_time

    test_accuracy = 100 * test_correct / test_total
    print(f"\n✅ Test Accuracy: {test_accuracy:.2f}%")
    print(f"🏆 Best Validation Accuracy: {best_val_accuracy:.2f}%")

    # Display test time
    print(f"\n⏱️  Test Execution Time:")
    print(f"   • Total time: {test_duration:.3f} seconds")
    print(f"   • Time per sample: {(test_duration / test_total * 1000):.2f} ms")
    print(f"   • Samples processed: {test_total} samples")

    # Calculate FPS (Frames Per Second)
    fps = test_total / test_duration
    print(f"   • Processing speed: {fps:.1f} samples/second")

    # Decode predictions
    y_true = label_encoder.inverse_transform(all_true_labels)
    y_pred = label_encoder.inverse_transform(all_predictions)

    # Confusion matrix
    print("\n" + "=" * 60)
    print("📊 CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Hand Gesture Recognition (UNI-DIRECTIONAL GRU)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_unidirectional.png', dpi=100, bbox_inches='tight')
    plt.show()

    # Enhanced classification report with per-class accuracy
    print("\n" + "=" * 60)
    print("📈 ENHANCED CLASSIFICATION REPORT")
    print("=" * 60)

    # Calculate per-class accuracy from confusion matrix
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print(f"\n{'Gesture':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10}")
    print("-" * 55)

    # Get standard classification report
    report_dict = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)

    for i, gesture in enumerate(label_encoder.classes_):
        prec = report_dict[gesture]['precision'] * 100
        rec = report_dict[gesture]['recall'] * 100
        f1 = report_dict[gesture]['f1-score'] * 100
        acc = per_class_accuracy[i] * 100
        print(f"{gesture:<15} {prec:8.2f}%   {rec:8.2f}%   {f1:8.2f}%   {acc:8.2f}%")

    # Overall metrics
    print("-" * 55)
    print(f"{'OVERALL':<15} {report_dict['macro avg']['precision']*100:8.2f}%   "
          f"{report_dict['macro avg']['recall']*100:8.2f}%   "
          f"{report_dict['macro avg']['f1-score']*100:8.2f}%   "
          f"{test_accuracy:8.2f}%")

    # Create a summary table
    print(f"\n📋 Summary of Per-Class Accuracy:")
    print("-" * 40)
    for i, gesture in enumerate(label_encoder.classes_):
        accuracy = per_class_accuracy[i] * 100
        print(f"{i+1}. {gesture:<10}: {accuracy:6.2f}% correct")

    # Training history plot
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'config': {
            'time_steps': Config.TIME_POINTS,
            'input_features': Config.FEATURES,
            'num_classes': Config.CLASSES,
            'gru_hidden_size': Config.GRU_HIDDEN_SIZE,
            'gru_bidirectional': False,  # UNI-DIRECTIONAL
            'kan_grid': Config.KAN_GRID_SIZE,
            'kan_spline_order': Config.KAN_SPLINE_ORDER,
            'has_linear_layer': False  # ✅ توضيح عدم وجود Linear
        },
        'performance': {
            'test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_accuracy,
            'per_class_accuracy': {gesture: float(acc) for gesture, acc in zip(label_encoder.classes_, per_class_accuracy)},
            'test_duration_seconds': test_duration
        }
    }, Config.MODEL_SAVE_PATH)

    print(f"\n💾 Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"✅ Model type: KAN only (NO Linear classifier) - Uni-Directional")

    return model, label_encoder


# ==================== PREDICTION FUNCTION ====================
def predict_single_gesture(model, label_encoder, processor, file_path, device='cpu'):
    """Predict gesture for a single file"""
    sequence, true_gesture = processor.load_single_file(file_path)

    if sequence is None:
        print(f"❌ Could not load file: {file_path}")
        return None, 0.0, None

    model.eval()
    input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(output, dim=1).item()
        confidence = torch.max(probabilities).item()

    predicted_gesture = label_encoder.inverse_transform([predicted_idx])[0]

    return predicted_gesture, confidence, true_gesture


# ==================== MAIN FUNCTION ====================
def main():
    print("=" * 70)
    print("🤖 HAND GESTURE RECOGNITION (UNI-DIRECTIONAL GRU)")
    print("=" * 70)
    print("\n📋 Model Architecture: UNI-DIRECTIONAL GRU (128 hidden) → KAN")
    print("✅ KAN outputs 7 classes directly (NO Linear classifier)")
    print(f"📁 Data folder: '{Config.DATA_FOLDER}'")

    print("\n🎯 Available Options:")
    print("   1. Train model (UNI-DIRECTIONAL GRU → KAN)")
    print("   2. Test single file")
    print("   3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        print("\n" + "=" * 60)
        print("🚀 TRAINING MODEL WITH UNI-DIRECTIONAL GRU")
        print("=" * 60)
        model, label_encoder = train_hand_gesture_model()

        if model is not None:
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETE!")
            print("=" * 60)
            print("📊 Files created:")
            print(f"   • {Config.MODEL_SAVE_PATH} - Trained model")
            print(f"   • training_history_unidirectional.png - Training plots")
            print(f"   • confusion_matrix_unidirectional.png - Confusion matrix")

    elif choice == '2':
        model_path = Config.MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            print(f"\n❌ No trained model found at '{model_path}'")
            print("💡 Please train a model first (option 1)")
            return

        print("\n" + "=" * 60)
        print("🧪 TESTING SINGLE FILE")
        print("=" * 60)

        # Load model
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # Create model
            model = HandGestureGRUKAN(
                input_features=checkpoint['config']['input_features'],
                num_classes=checkpoint['config']['num_classes']
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            label_encoder = checkpoint['label_encoder']
            processor = HandGestureDataProcessor()

            # Get file path
            file_path = input("Enter path to CSV file: ").strip()
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Measure prediction time
            start_time = time.time()
            predicted, confidence, true_gesture = predict_single_gesture(
                model, label_encoder, processor, file_path, device
            )
            end_time = time.time()
            prediction_time = end_time - start_time

            if predicted:
                print(f"\n🎯 Prediction Results:")
                print(f"   📁 File: {os.path.basename(file_path)}")
                print(f"   🤖 Predicted: {predicted}")
                print(f"   🎯 True: {true_gesture}")
                print(f"   💪 Confidence: {confidence:.2%}")
                print(f"   ✅ {'CORRECT' if predicted == true_gesture else '❌ WRONG'}")
                print(f"   ✅ KAN only (NO Linear classifier) - Uni-Directional")
                print(f"\n⏱️  Timing Information:")
                print(f"   • Prediction time: {prediction_time*1000:.2f} ms")
                print(f"   • Real-time factor: {prediction_time/(Config.TIME_POINTS/Config.SAMPLING_RATE):.3f}x")

        except Exception as e:
            print(f"❌ Error loading model: {e}")

    elif choice == '3':
        print("👋 Goodbye!")

    else:
        print("❌ Invalid choice!")


# ==================== RUN PROGRAM ====================
if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print(f"   1. Check if '{Config.DATA_FOLDER}' contains CSV files")
        print("   2. Ensure CSV files have correct columns (14 features)")
        print("   3. Install required packages: pip install torch numpy pandas matplotlib scikit-learn seaborn pykan")
        print("   4. For GPU support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")