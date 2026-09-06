"""
HAND GESTURE RECOGNITION - ODD FRAMES ONLY VERSION
COMPARISON: UNI-DIRECTIONAL GRU → KAN  vs  UNI-DIRECTIONAL GRU → MLP
Both models output classes directly (no separate Linear layer)
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
from kan import KAN
import glob
import os
import warnings
import seaborn as sns
import time

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    TIME_POINTS = 180  # ✅ 180 إطار فردي
    FEATURES = 14
    CLASSES = 7
    SAMPLING_RATE = 45

    # KAN parameters
    KAN_GRID_SIZE = 5
    KAN_SPLINE_ORDER = 3

    # GRU parameters
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2

    # Training parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    EPOCHS = 100
    DROPOUT_RATE = 0.3

    DATA_FOLDER = "my_hand_data"
    MODEL_SAVE_PATH_KAN = "hand_gesture_gru_kan_odd_frames.pth"
    MODEL_SAVE_PATH_MLP = "hand_gesture_gru_mlp_odd_frames.pth"


# ==================== MODEL 1: GRU → KAN (modified to output classes directly) ====================
class HandGestureGRUKAN(nn.Module):
    """GRU → KAN, KAN outputs class logits directly (no separate classifier)"""

    def __init__(self, time_steps=180, input_features=14, num_classes=7):
        super().__init__()

        print("\n" + "=" * 60)
        print("🧠 BUILDING MODEL: UNI-DIRECTIONAL GRU → KAN (direct output)")
        print("✅ ODD FRAMES VERSION: 180 frames (1, 3, 5, ..., 359)")
        print("=" * 60)

        self.time_steps = time_steps
        self.input_features = input_features

        # ✅ STEP 1: UNI-DIRECTIONAL GRU
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=Config.DROPOUT_RATE if Config.GRU_NUM_LAYERS > 1 else 0
        )

        self.gru_output_size = Config.GRU_HIDDEN_SIZE
        print(f"   • GRU Output Size: {self.gru_output_size}")

        self.attention = nn.MultiheadAttention(
            embed_dim=self.gru_output_size,
            num_heads=4,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )

        # ✅ STEP 2: KAN (outputs num_classes directly)
        print(f"   • KAN Layer: {self.gru_output_size} → 64 → 32 → {num_classes}")
        self.kan = KAN(
            width=[self.gru_output_size, 64, 32, num_classes],  # مباشرة 7
            grid=Config.KAN_GRID_SIZE,
            k=Config.KAN_SPLINE_ORDER,
            seed=42
        )

        self.layer_norm = nn.LayerNorm(input_features)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.bn = nn.BatchNorm1d(self.gru_output_size)

        print("✅ Model created successfully!")

    def forward(self, x):
        x = self.layer_norm(x)
        gru_out, _ = self.gru(x)
        attended, _ = self.attention(gru_out, gru_out, gru_out)
        context = attended[:, -1, :]
        context = self.bn(context)
        context = self.dropout(context)
        output = self.kan(context)  # shape: [batch, num_classes]
        return output


# ==================== MODEL 2: GRU → MLP (modified to output classes directly) ====================
class HandGestureGRUMLP(nn.Module):
    """GRU → MLP, MLP outputs class logits directly (no separate classifier)"""

    def __init__(self, time_steps=180, input_features=14, num_classes=7):
        super().__init__()

        print("\n" + "=" * 60)
        print("🧠 BUILDING MODEL: UNI-DIRECTIONAL GRU → MLP (direct output)")
        print("✅ ODD FRAMES VERSION: 180 frames (1, 3, 5, ..., 359)")
        print("=" * 60)

        self.time_steps = time_steps
        self.input_features = input_features

        # ✅ STEP 1: UNI-DIRECTIONAL GRU (same as KAN model)
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=Config.DROPOUT_RATE if Config.GRU_NUM_LAYERS > 1 else 0
        )

        self.gru_output_size = Config.GRU_HIDDEN_SIZE
        print(f"   • GRU Output Size: {self.gru_output_size}")

        self.attention = nn.MultiheadAttention(
            embed_dim=self.gru_output_size,
            num_heads=4,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )

        # ✅ STEP 2: MLP (outputs num_classes directly)
        print(f"   • MLP Layer: {self.gru_output_size} → 64 → 32 → {num_classes} (ReLU)")
        self.mlp = nn.Sequential(
            nn.Linear(self.gru_output_size, 64),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # آخر طبقة تنتج عدد الفئات مباشرة
        )

        self.layer_norm = nn.LayerNorm(input_features)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.bn = nn.BatchNorm1d(self.gru_output_size)

        print("✅ Baseline MLP model created successfully!")

    def forward(self, x):
        x = self.layer_norm(x)
        gru_out, _ = self.gru(x)
        attended, _ = self.attention(gru_out, gru_out, gru_out)
        context = attended[:, -1, :]
        context = self.bn(context)
        context = self.dropout(context)
        output = self.mlp(context)  # shape: [batch, num_classes]
        return output


# ==================== DATA PROCESSOR (unchanged) ====================
class HandGestureDataProcessor:
    def __init__(self, target_length=180):
        self.target_length = target_length
        self.required_features = [
            'palm_x', 'palm_y', 'palm_z',
            'palm_velocity_x', 'palm_velocity_y', 'palm_velocity_z',
            'grab_strength', 'pinch_strength', 'pinch_distance',
            'thumb_extended', 'index_extended', 'middle_extended',
            'ring_extended', 'pinky_extended'
        ]

        self.gesture_keywords = {
            'open': 'open', 'close': 'close', 'grab': 'grab',
            'push': 'push', 'pull': 'pull', 'raise': 'raise', 'lower': 'lower'
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

            for feature in self.required_features:
                if feature not in df.columns:
                    df[feature] = 0.0

            max_frames_needed = 2 * self.target_length
            available_frames = min(len(df), max_frames_needed)
            odd_indices = list(range(0, available_frames, 2))

            if len(odd_indices) == 0:
                return None, None

            feature_data = df.iloc[odd_indices][self.required_features].values.astype(np.float32)

            if len(feature_data) < self.target_length:
                padding = np.zeros((self.target_length - len(feature_data), len(self.required_features)))
                feature_data = np.vstack([feature_data, padding])
            elif len(feature_data) > self.target_length:
                feature_data = feature_data[:self.target_length]

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

        print(f"\n✅ Successfully loaded data: {len(X)} sequences, shape: {X.shape}")
        return X, y, file_paths


class HandGestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(label_encoder.transform(y))
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== Unified Training Function ====================
def run_training(model_type='kan'):
    """
    model_type: 'kan' or 'mlp'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print(f"📁 LOADING DATA - MODEL TYPE: {model_type.upper()}")
    print("=" * 60)

    processor = HandGestureDataProcessor(target_length=Config.TIME_POINTS)

    if not os.path.exists(Config.DATA_FOLDER):
        print(f"❌ Data folder '{Config.DATA_FOLDER}' not found!")
        return None, None

    X, y, file_paths = processor.load_all_files(Config.DATA_FOLDER)

    if len(X) == 0:
        print(f"❌ No valid data found in '{Config.DATA_FOLDER}'")
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

    # Create model based on type
    if model_type == 'kan':
        model = HandGestureGRUKAN(
            time_steps=Config.TIME_POINTS,
            input_features=Config.FEATURES,
            num_classes=Config.CLASSES
        )
        save_path = Config.MODEL_SAVE_PATH_KAN
        model_name = "GRU-KAN"
    else:  # mlp
        model = HandGestureGRUMLP(
            time_steps=Config.TIME_POINTS,
            input_features=Config.FEATURES,
            num_classes=Config.CLASSES
        )
        save_path = Config.MODEL_SAVE_PATH_MLP
        model_name = "GRU-MLP"

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Information:")
    print(f"   • Model: {model_name}")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Frame Selection: Odd frames only (180 frames)")

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

    print(f"\n🚀 Starting training for {Config.EPOCHS} epochs ({model_name})...")

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

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'\n📈 Epoch [{epoch + 1:3d}/{Config.EPOCHS}] ({model_name})')
            print(f'   📉 Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%')
            print(f'   📊 Val Loss:   {avg_val_loss:.4f}, Acc:   {val_accuracy:.2f}%')
            print(f'   🏆 Best Val Acc: {best_val_accuracy:.2f}%')

        if patience_counter >= 15:
            print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ==================== TESTING WITH TIMING ====================
    print("\n" + "=" * 60)
    print(f"🧪 TESTING MODEL ({model_name})")
    print("=" * 60)

    model.eval()
    test_correct, test_total = 0, 0
    all_predictions, all_true_labels = [], []

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

    end_time = time.time()
    test_duration = end_time - start_time

    test_accuracy = 100 * test_correct / test_total
    time_per_sample_ms = (test_duration / test_total) * 1000

    print(f"\n✅ Test Accuracy: {test_accuracy:.2f}%")
    print(f"🏆 Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"\n⏱️  Performance Metrics ({model_name}):")
    print(f"   • Total test time: {test_duration:.3f} seconds")
    print(f"   • Time per sample: {time_per_sample_ms:.2f} ms")
    print(f"   • Samples processed: {test_total} samples")
    print(f"   • Parameters: {total_params:,}")

    # Confusion matrix
    y_true = label_encoder.inverse_transform(all_true_labels)
    y_pred = label_encoder.inverse_transform(all_predictions)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name} (Odd Frames)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}_odd_frames.png', dpi=100, bbox_inches='tight')
    plt.show()

    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print(f"\n📋 Per-Class Accuracy ({model_name}):")
    for i, gesture in enumerate(label_encoder.classes_):
        print(f"   {gesture:<10}: {per_class_accuracy[i]*100:6.2f}%")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'config': {
            'time_steps': Config.TIME_POINTS,
            'input_features': Config.FEATURES,
            'num_classes': Config.CLASSES,
            'model_type': model_type,
            'frame_type': 'odd_frames_only'
        },
        'performance': {
            'test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_accuracy,
            'time_per_sample_ms': time_per_sample_ms,
            'parameters': total_params,
            'per_class_accuracy': {gesture: float(acc) for gesture, acc in zip(label_encoder.classes_, per_class_accuracy)}
        }
    }, save_path)

    print(f"\n💾 Model saved to: {save_path}")
    print(f"📊 Key Results for Table 6:")
    print(f"   • Accuracy: {test_accuracy:.2f}%")
    print(f"   • Latency: {time_per_sample_ms:.2f} ms")
    print(f"   • Parameters: {total_params:,}")

    return model, label_encoder


# ==================== Loading and Evaluation Functions ====================

def load_model(model_path: str, model_type: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    label_encoder = checkpoint['label_encoder']

    if model_type == 'kan':
        model = HandGestureGRUKAN(
            time_steps=config['time_steps'],
            input_features=config['input_features'],
            num_classes=config['num_classes']
        )
    else:  # mlp
        model = HandGestureGRUMLP(
            time_steps=config['time_steps'],
            input_features=config['input_features'],
            num_classes=config['num_classes']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, label_encoder, config


def evaluate_models_on_path(input_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 70)
    print(f"🧪 Evaluating both models on: {input_path}")
    print("=" * 70)

    if not os.path.exists(Config.MODEL_SAVE_PATH_KAN):
        print(f"❌ KAN model file not found: {Config.MODEL_SAVE_PATH_KAN}")
        return
    if not os.path.exists(Config.MODEL_SAVE_PATH_MLP):
        print(f"❌ MLP model file not found: {Config.MODEL_SAVE_PATH_MLP}")
        return

    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.csv"))
        if not files:
            print(f"❌ No CSV files found in folder: {input_path}")
            return
    else:
        print(f"❌ Path does not exist: {input_path}")
        return

    print(f"\n📁 Found {len(files)} CSV file(s).")

    print("\n📥 Loading models...")
    model_kan, label_encoder_kan, _ = load_model(Config.MODEL_SAVE_PATH_KAN, 'kan', device)
    model_mlp, label_encoder_mlp, _ = load_model(Config.MODEL_SAVE_PATH_MLP, 'mlp', device)

    if not np.array_equal(label_encoder_kan.classes_, label_encoder_mlp.classes_):
        print("⚠️  Warning: Label encoders do not match. Using KAN's encoder.")
        label_encoder = label_encoder_kan
    else:
        label_encoder = label_encoder_kan

    processor = HandGestureDataProcessor(target_length=Config.TIME_POINTS)

    sequences = []
    true_labels = []
    for file_path in files:
        seq, label = processor.load_single_file(file_path)
        if seq is not None and label is not None:
            sequences.append(seq)
            true_labels.append(label)
        else:
            print(f"⚠️  Skipping file (invalid data): {os.path.basename(file_path)}")

    if not sequences:
        print("❌ No valid data could be loaded.")
        return

    X = np.array(sequences, dtype=np.float32)
    y_true = np.array(true_labels)

    try:
        y_true_encoded = label_encoder.transform(y_true)
    except ValueError as e:
        print(f"❌ Label transformation error: {e}")
        print("   Make sure the labels in the files match the training labels.")
        return

    print(f"✅ Loaded {len(X)} samples.")

    X_tensor = torch.FloatTensor(X).to(device)

    # KAN evaluation
    print("\n" + "-" * 50)
    print("🔍 Evaluating GRU→KAN model ...")
    model_kan.eval()
    total_kan = len(X_tensor)

    start_time = time.time()
    with torch.no_grad():
        outputs_kan = model_kan(X_tensor)
    end_time = time.time()
    time_kan = end_time - start_time
    time_per_sample_kan = (time_kan / total_kan) * 1000

    _, preds_kan = torch.max(outputs_kan, 1)
    correct_kan = (preds_kan == torch.tensor(y_true_encoded, device=device)).sum().item()
    accuracy_kan = 100 * correct_kan / total_kan

    print(f"   ✅ Accuracy: {accuracy_kan:.2f}%")
    print(f"   ⏱️  Total time: {time_kan:.3f} seconds")
    print(f"   ⏱️  Time per sample: {time_per_sample_kan:.2f} ms")

    # MLP evaluation
    print("\n" + "-" * 50)
    print("🔍 Evaluating GRU→MLP model ...")
    model_mlp.eval()
    total_mlp = len(X_tensor)

    start_time = time.time()
    with torch.no_grad():
        outputs_mlp = model_mlp(X_tensor)
    end_time = time.time()
    time_mlp = end_time - start_time
    time_per_sample_mlp = (time_mlp / total_mlp) * 1000

    _, preds_mlp = torch.max(outputs_mlp, 1)
    correct_mlp = (preds_mlp == torch.tensor(y_true_encoded, device=device)).sum().item()
    accuracy_mlp = 100 * correct_mlp / total_mlp

    print(f"   ✅ Accuracy: {accuracy_mlp:.2f}%")
    print(f"   ⏱️  Total time: {time_mlp:.3f} seconds")
    print(f"   ⏱️  Time per sample: {time_per_sample_mlp:.2f} ms")

    # Comparison table
    print("\n" + "=" * 70)
    print("📊 Comparison Results on the Specified Files")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy (%)':<15} {'Time (ms/sample)':<20}")
    print("-" * 55)
    print(f"{'GRU → KAN':<20} {accuracy_kan:<15.2f} {time_per_sample_kan:<20.2f}")
    print(f"{'GRU → MLP':<20} {accuracy_mlp:<15.2f} {time_per_sample_mlp:<20.2f}")

    # Confusion matrices
    y_true_labels = label_encoder.inverse_transform(y_true_encoded)
    y_pred_kan = label_encoder.inverse_transform(preds_kan.cpu().numpy())
    y_pred_mlp = label_encoder.inverse_transform(preds_mlp.cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, title in zip(axes, [y_pred_kan, y_pred_mlp], ['GRU→KAN', 'GRU→MLP']):
        cm = confusion_matrix(y_true_labels, y_pred, labels=label_encoder.classes_)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        ax.set_title(f'Confusion Matrix - {title}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('comparison_confusion_matrices.png', dpi=100)
    plt.show()

    print("\n💾 Confusion matrices saved as 'comparison_confusion_matrices.png'")
    print("🏁 Evaluation complete.")


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("🤖 HAND GESTURE RECOGNITION - ODD FRAMES VERSION")
    print("=" * 70)

    print("\n🎯 Available options:")
    print("   1. Train GRU-KAN model (proposed)")
    print("   4. Train GRU-MLP model (baseline for comparison)")
    print("   2. Evaluate both models on a specific folder or a single CSV file")
    print("   5. Test a single file (old option)")
    print("   3. Exit")

    choice = input("\nEnter your choice (1, 2, 3, 4, or 5): ").strip()

    if choice == '1':
        run_training(model_type='kan')
    elif choice == '4':
        print("\n" + "=" * 60)
        print("🚀 Training baseline: GRU-MLP (for comparison)")
        print("=" * 60)
        run_training(model_type='mlp')
    elif choice == '2':
        path = input("Enter the full path to a folder or a CSV file: ").strip()
        if not os.path.exists(path):
            print(f"❌ Path '{path}' does not exist.")
        else:
            evaluate_models_on_path(path)
    elif choice == '5':
        model_path = Config.MODEL_SAVE_PATH_KAN
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        file_path = input("Enter path to CSV file: ").strip()
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, label_encoder, _ = load_model(model_path, 'kan', device)
        processor = HandGestureDataProcessor(target_length=Config.TIME_POINTS)
        seq, true_label = processor.load_single_file(file_path)
        if seq is None:
            print("❌ Could not load file.")
            return
        X = torch.FloatTensor(seq).unsqueeze(0).to(device)
        start = time.time()
        with torch.no_grad():
            out = model(X)
            prob = torch.softmax(out, dim=1)
            pred_idx = torch.argmax(out, dim=1).item()
            confidence = torch.max(prob).item()
        end = time.time()
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        print(f"\n🎯 Prediction:")
        print(f"   File: {os.path.basename(file_path)}")
        print(f"   Predicted: {pred_label}")
        print(f"   True: {true_label}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Time: {(end-start)*1000:.2f} ms")
        print(f"   {'✅ CORRECT' if pred_label == true_label else '❌ WRONG'}")
    elif choice == '3':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")