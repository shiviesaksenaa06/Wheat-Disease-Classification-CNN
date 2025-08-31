# Disease Classification in Wheat from Images (CNN + Transfer Learning)

A compact, reproducible repo for classifying wheat leaf diseases from images using Convolutional Neural Networks (CNNs). It compares transfer-learned baselines (VGG16, MobileNet, ResNet50) against a lightweight custom CNN designed for near real-time inference on modest hardware.

---

## 🔍 Problem & Motivation
Wheat diseases (e.g., rusts, tan spot) reduce yields globally. Early, low-cost detection can help farmers intervene promptly. This project builds an image classifier that distinguishes **Healthy**, **Yellow (Stripe) Rust**, **Brown (Leaf) Rust**, and **Tan Spot** directly from leaf photos. The approach emphasizes:
- **Transfer learning** for strong accuracy with smaller labeled datasets  
- **Lightweight inference** suitable for edge/mobile deployment  
- **Clear evaluation** across accuracy, precision, recall, F1, and latency

---
## 🧠 Methods

### Models
- **Baselines (transfer learning):** VGG16, MobileNet, ResNet50 (ImageNet weights; final layers re-trained for 4 classes)
- **Proposed model:** Minimal-layer custom CNN optimized for fast training and low latency while maintaining competitive accuracy

### Data Pipeline (high level)
1. **Data sources:** Public wheat-leaf image datasets consolidated for **Healthy**, **Yellow Rust**, **Brown Rust**, **Tan Spot** classes (balanced splits).  
2. **Preprocessing:** Resize to `224×224`, standardize channels; model-specific preprocessing (e.g., RGB→BGR for VGG/ResNet).  
3. **Augmentation:** Light transformations to reduce overfitting and improve generalization.  
4. **Split:** 70% train / 30% test.  
5. **Training:** Adam optimizer, cross-entropy loss, batch size 32, ~10 epochs (per model).  
6. **Evaluation:** Accuracy, Precision, Recall, F1; latency and parameter counts for practicality.

---

## 📊 Key Results (test set)

| Model     | Accuracy | Precision | Recall | F1   |
|-----------|---------:|----------:|-------:|-----:|
| VGG16     | 0.9381   | 0.9418    | 0.9381 | 0.9372 |
| MobileNet | 0.9545   | 0.9456    | 0.9487 | 0.9472 |
| ResNet50  | 0.8749   | 0.9022    | 0.8749 | 0.8748 |
| **Proposed CNN** | **0.9449** | **0.9471** | **0.9449** | **0.9445** |

**Efficiency (illustrative averages):**
- Training time: Proposed CNN ≈ **5 min**; MobileNet ≈ 30 min; VGG16 ≈ 100 min; ResNet50 ≈ 60 min  
- Inference: Proposed CNN ≈ **0.038 s/image**; MobileNet ≈ 0.050 s; VGG16 ≈ 0.560 s; ResNet50 ≈ 0.206 s

---

## 🚀 Quickstart

### 1) Environment
- Python 3.9+  
- Suggested packages:
  - `tensorflow` / `tensorflow-gpu`
  - `keras`
  - `numpy`, `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `opencv-python` (optional, for image utils)

Install:
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib opencv-python
```

### 2) Data
- Place class-organized images under a root like:
```
data/
  train/
    healthy/
    yellow_rust/
    brown_rust/
    tan_spot/
  test/
    healthy/
    yellow_rust/
    brown_rust/
    tan_spot/
```
- Or adapt the notebook’s data-loading cell to your paths.  

### 3) Run the Notebook
Open `Major Implementation.ipynb` and execute cells in order:
1. **Config & imports**
2. **Load & preprocess data**
3. **Choose model** (VGG16 / MobileNet / ResNet50 / Proposed)
4. **Train**
5. **Evaluate** (metrics + confusion matrix)
6. **Infer on new images**

---

## 🧪 Reproducing Experiments

- Keep image size **224×224** and batch size **32**.  
- Use **Adam** with default LR (unless you’re fine-tuning backbone layers).  
- Train for **~10 epochs** per model as in the study; adjust for convergence if your data differs.  
- For transfer learning: **freeze feature extractor** layers, **train only classifier head** first; optional fine-tuning later.

