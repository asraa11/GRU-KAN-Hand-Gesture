GRU-KAN: Hybrid Recurrent–Kolmogorov–Arnold Network for 3D Hand Gesture Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository contains the official implementation of the paper:

> **"GRU-KAN: A Hybrid Recurrent–Kolmogorov–Arnold Network for Low-Latency 3D Hand Gesture Classification using Direct Kinematic Features from Skeletal Data"**  
> *International Journal of Intelligent Engineering and Systems (IJIES), 2026.*

We propose a lightweight framework that combines **Gated Recurrent Units (GRU)** for temporal encoding with **Kolmogorov–Arnold Networks (KAN)** for nonlinear classification. The system processes raw 3D skeletal streams from an Ultraleap IR 170 camera, extracts 14 kinematic descriptors, and achieves state-of-the-art accuracy (up to 90.0%, ranging from 87.14% to 90.0% for the best KAN variants) with a low CPU inference latency of 57.06 ms for the best-performing model (att1dir180).



---

## 📁 Repository Structure & Model Files

The root directory contains exactly **9 model scripts**:

| File Name | Model Variant |
| :--- | :--- |
| `att1dir180.py` | GRU-KAN + Attention + Unidirectional + 180 frames |
| `att1dir360.py` | GRU-KAN + Attention + Unidirectional + 360 frames |
| `att2dir180.py` | GRU-KAN + Attention + Bidirectional + 180 frames |
| `att2dir360.py` | GRU-KAN + Attention + Bidirectional + 360 frames |
| `1dir180.py` | GRU-KAN *without* Attention + Unidirectional + 180 frames |
| `1dir360.py` | GRU-KAN *without* Attention + Unidirectional + 360 frames |
| `2dir180.py` | GRU-KAN *without* Attention + Bidirectional + 180 frames |
| `withoutattention2dir360.py` | GRU-KAN *without* Attention + Bidirectional + 360 frames |
| `att1dir180mlp.py` | **GRU-MLP** (MLP head instead of KAN) + Attention + Unidirectional + 180 frames *(Baseline)* |

> **Naming Convention**:  
> - `att` = with Multi-Head Attention, `1dir`/`2dir` = Unidirectional/Bidirectional GRU.  
> - `180`/`360` = number of frames per sequence (180 uses odd-frame decimation from 360 fps raw data).  
> - `withoutattention` / no prefix = without Attention mechanism.

---

## 📊 Dataset

The dataset is included in this repository under the `data/` folder. It consists of:

- **Total sequences**: 350
- **Gestures**: 7 (open, close, grab, push, pull, raise, lower)
- **Capture rate**: 90 fps
- **Duration**: 4 seconds per sequence (360 frames)
- **Participants**: 1 participant, using **both hands** (right and left)
- **Repetitions**: 50 per gesture (25 per hand)

### Data Format
Each CSV file contains 14 feature columns extracted from the raw skeletal data:
- **Positional (3D)**: `palm_x`, `palm_y`, `palm_z`
- **Kinematic (3D)**: `palm_velocity_x`, `palm_velocity_y`, `palm_velocity_z`
- **Interaction & Morphology (8)**: `grab_strength`, `pinch_strength`, `pinch_distance`, `thumb_extended`, `index_extended`, `middle_extended`, `ring_extended`, `pinky_extended`

---

## 🧠 Model Architecture

The GRU-KAN framework follows a deterministic feed-forward pipeline:

1. **Input**: `X ∈ ℝ^(B × T × 14)` (Batch, Time steps, Features)
2. **Layer Normalization**
3. **GRU Encoder**: 2 layers, hidden size = 128 (Unidirectional or Bidirectional)
4. **Multi-Head Attention**: 4 heads (optional, configurable)
5. **Pooling**: Extracts final hidden state `h_T ∈ ℝ^(B × H)`
6. **Batch Normalization & Dropout** (rate = 0.3)
7. **KAN Classifier**: `[H, 64, 32,7]` with learnable B-spline activations
8. **Softmax**: Final gesture probabilities

### Architectural Variants
We evaluate **8 configurations** varying:
- **Directionality**: Unidirectional (1-dir) vs. Bidirectional (2-dir)
- **Temporal Window**: Full sequence (360 frames) vs. Odd-frame decimation (180 frames)
- **Attention**: With attention (att) vs. Without attention (watt)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/GRU-KAN-Hand-Gesture.git
cd GRU-KAN-Hand-Gesture
2. Install Dependencies
bash
pip install -r requirements.txt
(Ensure you have Python 3.8+)

3. Prepare the Dataset
The dataset is already provided in the data/ folder. If you wish to use your own data, place your CSV files in data/ with the same 14-feature format.

4. Train a Model
To train the GRU-KAN model on odd frames (180) with attention:

bash
python train.py --config configs/att1dir180.yaml
To train the GRU-MLP baseline (for ablation study):

bash
python train.py --config configs/gru_mlp.yaml
5. Evaluate and Measure Latency
bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir data/
📈 Results Summary
Model	Accuracy	Latency (per seq)	Parameters
GRU-MLP (Baseline)	85.71%	12.77 ms	231,267
GRU-KAN (att1dir180)	87.14%	55.60 ms	364,291
GRU-KAN (watt2dir180)	90.0%	88.8 ms	668,252
Note: All latency values are measured on a CPU-only Intel Core i7-12700K workstation. The reported values correspond to the neural-network inference step after the sequence has been acquired.

📁 Repository Structure
text
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Full dataset (CSV files)
│   ├── open_seq1.csv
│   ├── close_seq1.csv
│   └── ...
├── models/                   # Model definitions
│   ├── gru_kan.py
│   ├── gru_mlp.py
│   └── attention.py
├── preprocessing/            # Feature extraction & normalization
│   ├── feature_extractor.py
│   └── data_loader.py
├── training/                 # Training scripts
│   ├── train.py
│   └── configs/              # YAML configs for 8 variants
├── evaluation/               # Evaluation & latency measurement
│   ├── evaluate.py
│   └── latency_measure.py
├── splits/                   # Fixed train/val/test indices
│   └── indices_seed42.npy
└── checkpoints/              # Pretrained models (optional)
    └── best_att1dir180.pth
🛠️ Configuration (config.yaml)
All hyperparameters are fixed to ensure reproducibility:

Parameter	Value
Optimizer	Adam
Learning Rate	1e-3
Batch Size	16
GRU Layers	2
Hidden Size	128
Dropout	0.3
Patience (Early Stopping)	15 epochs
Max Epochs	100
📝 Citation
If you find this code useful in your research, please cite our paper:

bibtex
@article{fatih2026grukan,
  title={GRU-KAN: A Hybrid Recurrent–Kolmogorov–Arnold Network for Low-Latency 3D Hand Gesture Classification using Direct Kinematic Features from Skeletal Data},
  author={Fatih, Israa F. and Hacham, Wisam S. and Sabir, Muhannad K.},
  journal={International Journal of Intelligent Engineering and Systems},
  year={2026}
}
Or open an issue on GitHub.

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

text
