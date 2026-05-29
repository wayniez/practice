# STL-10 Image Classifier

A custom CNN with residual connections trained on the [STL-10](https://cs.stanford.edu/~acomacho/stl10/) dataset. The project covers the full pipeline: data loading with augmentation, model training with early stopping, pseudo-labeling on unlabeled data, and evaluation via a confusion matrix.

---

## Dataset

**STL-10** is a 10-class image recognition benchmark derived from ImageNet, designed for semi-supervised learning. Images are 96×96 RGB.

| Split | Samples |
|---|---|
| Train | 5,000 |
| Test | 8,000 |
| Unlabeled | 100,000 |

**Classes:** airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck

---

## Model Architecture

A lightweight custom CNN (`STL10Model`) built with residual blocks.

```
Input (3×96×96)
  └─ Conv2d(3→32, stride=2) + BN + ReLU + MaxPool2d
  └─ ResidualBlock(32)
  └─ Conv2d(32→64) + BN + ReLU
  └─ ResidualBlock(64)
  └─ Conv2d(64→128) + BN + ReLU
  └─ ResidualBlock(128)
  └─ AdaptiveAvgPool2d(1) + Flatten
  └─ Dropout(0.4)
  └─ Linear(128 → 10)
```

Each `ResidualBlock` is a two-layer conv stack with a skip connection:
```
x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
```

---

## Training

**Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)  
**Loss:** CrossEntropyLoss  
**Batch size:** 32  
**Early stopping:** patience = 8 epochs (monitors test loss)  
**Checkpoint:** best model saved to `model_params/best_model.pth`

### Data Augmentation (train only)

- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Random crop (96×96, padding=8)
- Color jitter (brightness, contrast, saturation ±0.2)
- Normalize to mean/std of 0.5

---

## Pseudo-Labeling (Semi-Supervised)

After initial supervised training, the model labels a subset of the 100,000 unlabeled images using a confidence threshold of **0.90**. High-confidence predictions are added to the training set and the model is fine-tuned on the combined dataset.

> **Result:** Pseudo-labeling did not improve accuracy in this experiment, likely because the model had already reached its capacity limit without transfer learning.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy (supervised) | ~69.7% |
| Test accuracy (+ LR scheduler) | ~69.8% |
| Test accuracy (+ pseudo-labels) | No improvement |

**Key observations from the confusion matrix:**
- The model reliably separates vehicles from animals.
- It struggles to distinguish visually similar animal classes (e.g., cat vs. dog, deer vs. horse) — a known limitation of CNNs trained from scratch on small datasets.

---

## Requirements

```
torch
torchvision
torchmetrics
matplotlib
seaborn
tqdm
```

Install with:
```bash
pip install torch torchvision torchmetrics matplotlib seaborn tqdm
```

---

## Usage

1. **Clone the repo and open the notebook:**
   ```bash
   jupyter notebook STL10_Model.ipynb
   ```

2. The STL-10 dataset will be **downloaded automatically** to `./data/` on first run.

3. Training uses GPU automatically if available (`cuda`), otherwise falls back to CPU.

4. The best checkpoint is saved to `model_params/best_model.pth` and reloaded for evaluation and pseudo-labeling.

---

## Project Structure

```
.
├── STL10_Model.ipynb       # Main notebook
├── model_params/
│   └── best_model.pth      # Saved checkpoint (created at runtime)
└── data/                   # STL-10 dataset (downloaded at runtime)
```

---

## Limitations & Future Work

- **Transfer learning** (e.g., fine-tuning a pretrained ResNet or EfficientNet) would likely push accuracy well above 70% on this dataset.
- **Self-supervised pretraining** (SimCLR, MoCo) on the unlabeled split before fine-tuning is a more principled way to leverage the 100K unlabeled images.
- The current pseudo-labeling approach is limited by the base model's accuracy ceiling — a stronger initial model would produce better pseudo-labels.
