"""
HAND GESTURE RECOGNITION
GRU-KAN vs GRU-MLP — FPHA 45 CLASSES, 63 FEATURES
VARIABLE-LENGTH VERSION (uses all real frames, no padding waste)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from kan import KAN
import glob
import os
import warnings
import seaborn as sns
import time

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    FEATURES = 63
    CLASSES = 45
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    KAN_GRID_SIZE = 5
    KAN_SPLINE_ORDER = 3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    EPOCHS = 100
    DROPOUT_RATE = 0.5



    DATA_FOLDER = "processed_45classes_63feat"
    MODEL_SAVE_PATH_KAN = "hand_gesture_gru_kan_45class_varlen.pth"
    MODEL_SAVE_PATH_MLP = "hand_gesture_gru_mlp_45class_varlen.pth"


# ==================== 45 فئة FPHA ====================
FPHA_45_CLASSES = [
    'open_juice_bottle', 'close_juice_bottle', 'pour_juice_bottle',
    'open_peanut_butter', 'close_peanut_butter', 'prick', 'sprinkle',
    'scoop_spoon', 'put_sugar', 'stir', 'open_milk', 'close_milk',
    'pour_milk', 'drink_mug', 'put_tea_bag', 'put_salt',
    'open_liquid_soap', 'close_liquid_soap', 'pour_liquid_soap',
    'wash_sponge', 'flip_sponge', 'scratch_sponge', 'squeeze_sponge',
    'open_soda_can', 'use_flash', 'write', 'tear_paper', 'squeeze_paper',
    'open_letter', 'take_letter_from_enveloppe', 'read_letter',
    'flip_pages', 'use_calculator', 'light_candle', 'charge_cell_phone',
    'unfold_glasses', 'clean_glasses', 'open_wallet', 'give_coin',
    'receive_coin', 'give_card', 'pour_wine', 'toast_wine',
    'handshake', 'high_five'
]

FEATURE_COLUMNS = []
for j in range(21):
    FEATURE_COLUMNS.extend([f'j{j}_x', f'j{j}_y', f'j{j}_z'])


# ==================== MODEL 1: GRU → KAN (Variable-Length) ====================
class HandGestureGRUKAN(nn.Module):
    def __init__(self, input_features=63, num_classes=45):
        super().__init__()

        print("\n" + "=" * 60)
        print("🧠 BUILDING MODEL: GRU → KAN (Variable-Length)")
        print("✅ FPHA 45-CLASS, 63 FEATURES")
        print("=" * 60)

        self.input_features = input_features

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

        print(f"   • KAN Layer: {self.gru_output_size} → 64 → 32 → {num_classes}")
        self.kan = KAN(
            width=[self.gru_output_size, 64, 32, num_classes],
            grid=Config.KAN_GRID_SIZE,
            k=Config.KAN_SPLINE_ORDER,
            seed=42
        )

        self.layer_norm = nn.LayerNorm(input_features)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.bn = nn.BatchNorm1d(self.gru_output_size)

        print("✅ Model created successfully!")

    def forward(self, x, lengths):
        """
        x: (B, T_max, F) — padded
        lengths: (B,) — actual lengths
        """
        # LayerNorm على الإدخال
        x = self.layer_norm(x)

        # ✅ pack_padded_sequence — يتجاهل الـ padding
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # GRU
        packed_out, _ = self.gru(packed)

        # ✅ unpack
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # ✅ attention mask — نمنع الانتباه للـ padding
        B, T_max, _ = gru_out.shape
        device = gru_out.device
        mask = torch.arange(T_max, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        # mask: (B, T_max), True = padding

        attended, _ = self.attention(gru_out, gru_out, gru_out, key_padding_mask=mask)

        # ✅ نأخذ آخر إطار حقيقي لكل تسلسل
        idx = (lengths - 1).long().to(attended.device)
        context = attended[torch.arange(B, device=attended.device), idx, :]

        context = self.bn(context)
        context = self.dropout(context)
        output = self.kan(context)
        return output


# ==================== MODEL 2: GRU → MLP (Variable-Length) ====================
class HandGestureGRUMLP(nn.Module):
    def __init__(self, input_features=63, num_classes=45):
        super().__init__()

        print("\n" + "=" * 60)
        print("🧠 BUILDING MODEL: GRU → MLP (Variable-Length)")
        print("✅ FPHA 45-CLASS, 63 FEATURES")
        print("=" * 60)

        self.input_features = input_features

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

        print(f"   • MLP Layer: {self.gru_output_size} → 64 → 32 → {num_classes}")
        self.mlp = nn.Sequential(
            nn.Linear(self.gru_output_size, 64),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.layer_norm = nn.LayerNorm(input_features)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.bn = nn.BatchNorm1d(self.gru_output_size)

        print("✅ Baseline MLP model created successfully!")

    def forward(self, x, lengths):
        x = self.layer_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        B, T_max, _ = gru_out.shape
        device = gru_out.device
        mask = torch.arange(T_max, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        attended, _ = self.attention(gru_out, gru_out, gru_out, key_padding_mask=mask)

        idx = (lengths - 1).long().to(attended.device)
        context = attended[torch.arange(B, device=attended.device), idx, :]

        context = self.bn(context)
        context = self.dropout(context)
        output = self.mlp(context)
        return output


# ==================== DATA PROCESSOR ====================
class HandGestureDataProcessor:
    def __init__(self):
        self.required_features = FEATURE_COLUMNS
        self.valid_gestures = FPHA_45_CLASSES

    def load_single_file(self, file_path: str) -> Tuple[np.ndarray, str]:
        try:
            df = pd.read_csv(file_path)

            parent_folder = os.path.basename(os.path.dirname(file_path)).lower()
            gesture_name = None
            for g in self.valid_gestures:
                if g.lower() == parent_folder:
                    gesture_name = g
                    break

            if gesture_name is None:
                return None, None

            for feature in self.required_features:
                if feature not in df.columns:
                    df[feature] = 0.0

            # ✅ لا padding ولا odd-decimation — نحتفظ بالطول الأصلي
            feature_data = df[self.required_features].values.astype(np.float32)

            # Z-score normalization
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

    def load_all_files(self, data_folder: str) -> Tuple[List[np.ndarray], np.ndarray]:
        print(f"📁 Loading data from: {data_folder}")
        csv_files = glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True)

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_folder}")

        print(f"Found {len(csv_files)} CSV files")

        all_sequences, all_labels = [], []
        lengths = []

        for file_path in csv_files:
            sequence, label = self.load_single_file(file_path)
            if sequence is not None and label is not None:
                all_sequences.append(sequence)
                all_labels.append(label)
                lengths.append(len(sequence))

        if not all_sequences:
            raise ValueError("No valid data loaded!")

        y = np.array(all_labels)
        lengths = np.array(lengths)

        print(f"\n✅ Successfully loaded {len(all_sequences)} sequences")
        print(f"📊 Length stats: min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}, median={np.median(lengths):.0f}")
        print(f"🎯 Unique classes: {len(np.unique(y))}")

        return all_sequences, y, lengths


class HandGestureDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], y: np.ndarray,
                 lengths: np.ndarray, label_encoder: LabelEncoder):
        self.sequences = sequences
        self.y = torch.LongTensor(label_encoder.transform(y))
        self.lengths = torch.LongTensor(lengths)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            self.lengths[idx],
            self.y[idx]
        )


def collate_variable_length(batch):
    """Custom collate — يضمّن كل batch حسب أطول تسلسل فيه"""
    sequences, lengths, labels = zip(*batch)
    # pad إلى أطول تسلسل في هذا batch فقط
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    return padded, lengths, labels


# ==================== Training Function ====================
def run_training(model_type='kan'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")

    print("\n" + "=" * 60)
    print(f"📁 LOADING DATA - MODEL: {model_type.upper()}")
    print("=" * 60)

    processor = HandGestureDataProcessor()

    if not os.path.exists(Config.DATA_FOLDER):
        print(f"❌ Data folder '{Config.DATA_FOLDER}' not found!")
        return None, None

    sequences, y, lengths = processor.load_all_files(Config.DATA_FOLDER)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n🎯 Number of classes: {len(label_encoder.classes_)}")

    # تقسيم مع الحفاظ على التوزيع
    indices = np.arange(len(sequences))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, stratify=y_encoded, random_state=42
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=0.15, stratify=y_encoded[idx_train], random_state=42
    )

    train_seqs = [sequences[i] for i in idx_train]
    val_seqs = [sequences[i] for i in idx_val]
    test_seqs = [sequences[i] for i in idx_test]

    train_lengths = lengths[idx_train]
    val_lengths = lengths[idx_val]
    test_lengths = lengths[idx_test]

    train_y = y_encoded[idx_train]
    val_y = y_encoded[idx_val]
    test_y = y_encoded[idx_test]

    print(f"\n📊 Data split:")
    print(f"   Training: {len(train_seqs)} sequences")
    print(f"   Validation: {len(val_seqs)} sequences")
    print(f"   Test: {len(test_seqs)} sequences")

    train_dataset = HandGestureDataset(train_seqs, label_encoder.inverse_transform(train_y), train_lengths, label_encoder)
    val_dataset = HandGestureDataset(val_seqs, label_encoder.inverse_transform(val_y), val_lengths, label_encoder)
    test_dataset = HandGestureDataset(test_seqs, label_encoder.inverse_transform(test_y), test_lengths, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=0, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=0, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=0, collate_fn=collate_variable_length)

    if model_type == 'kan':
        model = HandGestureGRUKAN(Config.FEATURES, Config.CLASSES)
        save_path = Config.MODEL_SAVE_PATH_KAN
        model_name = "GRU-KAN"
    else:
        model = HandGestureGRUMLP(Config.FEATURES, Config.CLASSES)
        save_path = Config.MODEL_SAVE_PATH_MLP
        model_name = "GRU-MLP"

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model: {model_name}")
    print(f"   • Parameters: {total_params:,}")
    print(f"   • Features: 63")
    print(f"   • Variable-length: YES (no padding waste)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_accuracy = 0
    best_model_state = None
    patience_counter = 0

    print(f"\n🚀 Training for {Config.EPOCHS} epochs ({model_name})...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_X, batch_lengths, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, batch_lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_X, batch_lengths, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_lengths = batch_lengths.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X, batch_lengths)
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

    if best_model_state:
        model.load_state_dict(best_model_state)

    # ==================== TESTING ====================
    print("\n" + "=" * 60)
    print(f"🧪 TESTING ({model_name})")
    print("=" * 60)

    model.eval()
    test_correct, test_total = 0, 0
    all_predictions, all_true_labels = [], []

    start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_lengths, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X, batch_lengths)
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
    print(f"\n⏱️  Metrics ({model_name}):")
    print(f"   • Time per sample: {time_per_sample_ms:.2f} ms")
    print(f"   • Parameters: {total_params:,}")

    y_true = label_encoder.inverse_transform(all_true_labels)
    y_pred = label_encoder.inverse_transform(all_predictions)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name} (FPHA 45 Classes, Variable-Length)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}_varlen.png', dpi=100, bbox_inches='tight')
    plt.show()

    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'config': {
            'input_features': Config.FEATURES,
            'num_classes': Config.CLASSES,
            'model_type': model_type,
            'padding': 'variable-length',
        },
        'performance': {
            'test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_accuracy,
            'time_per_sample_ms': time_per_sample_ms,
            'parameters': total_params,
        }
    }, save_path)

    print(f"\n💾 Model saved: {save_path}")
    print(f"📊 Final Results: Accuracy={test_accuracy:.2f}%, Latency={time_per_sample_ms:.2f} ms, Params={total_params:,}")

    return model, label_encoder


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("🤖 GRU-KAN vs GRU-MLP — FPHA 45 CLASSES (Variable-Length)")
    print("=" * 70)

    print("\n🎯 Options:")
    print("   1. Train GRU-KAN")
    print("   4. Train GRU-MLP")
    print("   3. Exit")

    choice = input("\nEnter your choice (1, 3, or 4): ").strip()

    if choice == '1':
        run_training(model_type='kan')
    elif choice == '4':
        run_training(model_type='mlp')
    elif choice == '3':
        print("👋 Goodbye!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")