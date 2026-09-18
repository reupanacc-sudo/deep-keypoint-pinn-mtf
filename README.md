# ⚡ 2D ResNet-Edge MTF Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

An open-source, high-speed deep learning optical quality inspection engine. Replaces legacy slanted-edge software with a **2D ResNet-Edge Convolutional Neural Network** for **sub-millisecond Edge-to-MTF regression** combined with a certified **ISO 12233 4-stage physics baseline (ESF → LSF → FFT → MTF)**.

---

## 🌟 Key Features

- **⚡ Sub-Millisecond MTF Regression**: Direct 2D convolutional regression from raw edge patches to continuous MTF curves in $< 3\text{ ms}$ on CPU ($< 0.3\text{ ms}$ on GPU).
- **🔬 4-Stage Physics Decomposition**: Real-time breakdown of Normalized ESF, Hann-windowed LSF bell curve, FFT magnitude spectrum, and bounded MTF curve ($0\%$ to $100\%$).
- **🛡️ Noise & Flare Immune**: Trained with Fourier optics loss constraints ($\mathcal{L}_{\text{PINN}}$) that enforce physical monotonicity ($\partial \text{MTF}/\partial f \le 0$) and eliminate high-frequency derivative noise spikes.
- **📷 Universal & Plug-and-Play**: Load **ANY image** format (`.png`, `.jpg`, `.bmp`, `.tif`), select ROIs interactively via click-and-drag, or use the built-in slanted edge auto-detector.
- **🖥️ Standalone Dark-Themed GUI**: Built-in visualizer with real-time curve plotting and CSV export.

---

## 🏗️ Model Architecture

```
                    Raw Slanted-Edge Crop (45 x 40)
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  In-Conv: 3x3 Conv + BN + ReLU│ (32 channels)
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  ResBlock 1: Stride 2 Conv    │ (64 channels)
                 ├───────────────────────────────┤
                 │  ResBlock 2: Stride 2 Conv    │ (128 channels)
                 ├───────────────────────────────┤
                 │  ResBlock 3: Stride 2 Conv    │ (128 channels)
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  Adaptive AvgPool + FC Head   │
                 ├───────────────────────────────┤
                 │  Sigmoid Output (0.0 to 1.0)  │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
             Full MTF Curve Across 64 Spatial Frequencies
```

### Physics-Informed Fourier Optics Loss ($\mathcal{L}_{\text{PINN}}$)

$$\mathcal{L} = \mathcal{L}_{\text{MSE}}(y_{\text{pred}}, y_{\text{true}}) + w_1 \underbrace{|MTF(0) - 1.0|^2}_{\text{DC Normalization}} + w_2 \underbrace{\left\|\text{ReLU}\left(\frac{\partial MTF}{\partial f}\right)\right\|^2}_{\text{Monotonicity Penalty}} + w_3 \underbrace{\left\|\frac{\partial^2 MTF}{\partial f^2}\right\|^2}_{\text{Smoothness}}$$

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/reupanacc-sudo/deep-keypoint-pinn-mtf.git
cd deep-keypoint-pinn-mtf
pip install -r requirements.txt
```

### 2. Launch the Desktop GUI

```bash
python demo_dual_engine_gui.py
```
*(Or double-click `run_dual_engine_gui.bat` on Windows)*

* **Open Image**: Load any camera image or edge patch.
* **Select ROI**: Click and drag a box around any slanted edge, or click **Auto-Detect Edges**.
* **Instant MTF**: View the predicted MTF50, line-rate latency, and 4-stage physics plots in real time.

---

## 🐍 Python API (3-Line Integration)

You can use the engine directly in your own Python scripts:

```python
import cv2
from openmtf import ResNetEdgeMTFEngine

# Initialize the engine
engine = ResNetEdgeMTFEngine()

# Load any slanted-edge crop
crop = cv2.imread("my_edge_crop.png", cv2.IMREAD_GRAYSCALE)

# Run sub-millisecond inference
results = engine.analyze_roi_full(crop, target_freq_lpmm=50.0)

print(f"Neural MTF:      {results['neural']['NeuralMTFPercent']:.2f}%")
print(f"Neural MTF50:    {results['neural']['NeuralMTF50']:.1f} lp/mm")
print(f"Classical MTF:   {results['classical']['FilteredMTF']*100:.2f}%")
print(f"Inference Time:  {results['neural']['InferenceTimeMs']} ms")
```

---

## 📊 Benchmarks & Training

### Run Performance & Noise Stress Tests
```bash
python benchmarks/benchmark_resnet_edge.py
```

### Train with Synthetic Optics Generator
```bash
python training/train_pinn_mtf.py
```

---

## 📁 Repository Structure

```text
├── models/
│   ├── pinn_mtf.py              # 2D ResNet-Edge model & Fourier loss
│   └── deep_keypoint.py         # Spatial soft-argmax keypoint detector
├── inference/
│   ├── engine.py                # Core sub-millisecond inference engine
│   └── dual_engine.py           # Universal wrapper
├── training/
│   ├── synthetic_edge_generator.py # Physical PSF & slanted edge synthesizer
│   └── train_pinn_mtf.py        # Model training pipeline
├── benchmarks/
│   └── benchmark_resnet_edge.py # Latency, accuracy & noise stress tests
├── checkpoints/
│   └── pinn_mtf_weights.pt      # Pre-trained neural weights
├── mtf_core.py                  # ISO 12233 4-stage physics pipeline
├── demo_dual_engine_gui.py      # Desktop inspection visualizer
├── app.py                       # GUI entrypoint
├── openmtf.py                   # Public Python SDK API
├── requirements.txt             # Dependencies
└── LICENSE                      # MIT License
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
