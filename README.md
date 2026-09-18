# 🚀 Deep Keypoint + PINN-MTF Dual Engine Inspector

An advanced AI-powered optical quality inspection system combining **Zero-Shot Deep Keypoint Localization** and a **Physics-Informed Neural Network (PINN-MTF)** with Classical **ISO 12233** validation for industrial camera module testing.

---

## 🏗️ System Architecture

```
                                  CAMERA MODULE RAW IMAGE
                                             │
                                             ▼
             ┌───────────────────────────────────────────────────────────────┐
             │  Stage 1: Deep Keypoint & Crosshair Subpixel Network          │
             │  • Auto-locates fiducial centers & crosshair intersection (X₀,Y₀)
             │  • Zero configuration required: Adapts to any camera chart!   │
             └───────────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  DUAL-ENGINE PARALLEL PIPELINE │
                             └──────┬─────────────────┬──────┘
                                    │                 │
            ┌───────────────────────┴──────┐   ┌──────┴────────────────────────┐
            │ Stage 2A: PINN-MTF Regressor │   │ Stage 2B: Classical ISO 12233 │
            │ • Direct Fourier Optics Loss │   │ • 4x Super-sampled ESF        │
            │ • Sub-millisecond inference  │   │ • Derivative & Windowed FFT   │
            │ • Resilient to noise & glare │   │ • Standard Baseline           │
            └───────────────────────┬──────┘   └──────┬────────────────────────┘
                                    │                 │
                                    ▼                 ▼
             ┌───────────────────────────────────────────────────────────────┐
             │  Stage 3: Consensus Validation & Optical Anomaly Detection     │
             │  • Real-time confidence score: |Neural - Classical| < 6%      │
             │  • Flags lens defects, glare, smudge, and sensor anomalies    │
             └───────────────────────────────────────────────────────────────┘
```

---

## 🔬 Physics-Informed Neural Network (PINN) Loss

The PINN model trains with an optical physics regularizer:

$$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{MSE}}(MTF_{\text{pred}}, MTF_{\text{true}}) + w_1 \underbrace{|MTF(0) - 1.0|^2}_{\text{DC Normalization}} + w_2 \underbrace{\left\|\text{ReLU}\left(\frac{\partial MTF}{\partial f}\right)\right\|^2}_{\text{Monotonicity Constraint}} + w_3 \underbrace{\left\|\frac{\partial^2 MTF}{\partial f^2}\right\|^2}_{\text{Curvature Smoothness}}$$

---

## 📁 Folder Structure

```
Deep Keypoint + PINN-MTF dual engine/
├── models/
│   ├── pinn_mtf.py          # ResNet-Edge PINN architecture & Fourier loss
│   └── deep_keypoint.py     # Spatial Soft-Argmax Keypoint locator
├── training/
│   ├── synthetic_edge_generator.py # Physics-accurate optical edge synthesizer
│   └── train_pinn_mtf.py    # Training & optimization pipeline
├── inference/
│   └── dual_engine.py       # Dual-Engine orchestrator & consensus validator
├── benchmarks/
│   └── benchmark_dual_engine.py # Production datasets & noise stress benchmarks
├── checkpoints/
│   └── pinn_mtf_weights.pt  # Trained PyTorch neural model checkpoint
├── configs/                 # Target profile presets (Bosch, Idneo, Melco)
├── demo_dual_engine_gui.py  # Modern interactive dark-theme desktop application
└── run_dual_engine_gui.bat  # 1-click Windows GUI launcher
```

---

## 🚀 How to Run

### 1. Launch the Dual-Engine Desktop GUI
Double click `run_dual_engine_gui.bat` or run:
```bash
python demo_dual_engine_gui.py
```

### 2. Run Comprehensive Benchmarks & Stress Tests
```bash
python benchmarks/benchmark_dual_engine.py
```

### 3. Re-Train the Physics-Informed Neural Network
```bash
python training/train_pinn_mtf.py
```
