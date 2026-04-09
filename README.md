# Brain Tumor Segmentation on MRI

Semantic segmentation of brain tumors on multi-modal MRI using deep neural networks.
Originally a BCS thesis project, now extended with modern architectures, PEFT methods,
and a full training pipeline.

## Setup

```bash
pip install -r requirements.txt
```

Set the data directory (default: `/notebooks/shared/data`):
```bash
export DATA_DIR=/path/to/your/data
```

Download the BraTS dataset (Medical Segmentation Decathlon Task01_BrainTumour).

## Training

### CLI Training (recommended)

```bash
# Full fine-tuning with SegResNet
python train.py --arch SegResNet --max_epochs 100 --batch_size 2 --lr 1e-4

# SwinUNETR with cosine schedule
python train.py --arch UNETR --scheduler cosine --precision 16-mixed

# MedNeXt (new architecture)
python train.py --arch MedNeXt --max_epochs 150 --optim adamw
```

### Resource-Constrained Training (PEFT)

```bash
# LoRA fine-tuning (~5-10% of parameters, ~3x less VRAM)
python train.py --arch UNETR --peft lora --lora_rank 4 --lr 1e-3

# Frozen encoder (train only decoder + head)
python train.py --arch SegResNet --peft freeze_encoder --lr 1e-3

# Adapter injection
python train.py --arch SegResNet --peft adapters --lr 5e-4
```

### Notebook Training (legacy)

Run `thesis.ipynb` — use the `Config` dataclass to configure experiments.

## Available Architectures

| Architecture | Source | Key Feature |
|---|---|---|
| SegResNet | MONAI | Residual encoder, fast training |
| SegResNetVAE | MONAI | VAE regularization branch |
| SwinUNETR | MONAI | Shifted-window transformer encoder |
| DynUNet (nnUNet) | MONAI | Self-configuring U-Net |
| UNet | MONAI | Classic 3D U-Net |
| AttentionUnet | MONAI | Attention gates on skip connections |
| ConvNeXt | Custom | 3D ConvNeXt with large kernels |
| MedNeXt | Custom | Medical ConvNeXt encoder-decoder |
| ResNet3dVAE | Custom | ResNet encoder + VAE branch |
| CustomnnUNet | Custom | Manual nnUNet with skip connections |

## Pre-trained Models

```python
from utils.hub import load_monai_bundle, load_hf_swinunetr_pretrained

# MONAI Model Zoo bundle
model = load_monai_bundle("brats_mri_segmentation")

# SwinUNETR with self-supervised pre-trained weights
model = load_hf_swinunetr_pretrained(in_channels=4, out_channels=3)
```

## Inference

```python
from utils.inference import sliding_window_predict, ensemble_predict
from utils.postprocess import postprocess_brats
from utils.tta import tta_predict

# Sliding window for full-resolution volumes
pred = sliding_window_predict(model, image, roi_size=(128, 128, 128), overlap=0.5)

# Test-time augmentation
pred = tta_predict(model, image)

# Multi-model ensemble + TTA
from utils.inference import ensemble_predict_with_tta
pred = ensemble_predict_with_tta([segresnet, swinunetr, mednext], image)

# Post-processing (remove small components, enforce TC/WT/ET hierarchy)
clean = postprocess_brats(pred, threshold=0.5, min_size_et=50)
```

## Uncertainty Estimation

```python
from utils.uncertainty import mc_dropout_predict, uncertainty_map

mean_pred, variance = mc_dropout_predict(model, image, n_samples=20)
unc = uncertainty_map(variance)  # per-voxel confidence
```

## Evaluation Metrics

```python
from utils.metrics import BraTSMetrics

metrics = BraTSMetrics()
results = metrics.compute(pred_binary, target_binary)
# {"TC_dice": 0.89, "TC_hd95": 3.2, "WT_dice": 0.91, "ET_dice": 0.82, ...}
```

## Project Structure

```
bt_segmentation/
├── train.py                    # Lightning CLI trainer
├── thesis.ipynb                # Original notebook (legacy)
├── requirements.txt
├── models/
│   ├── BaseModel.py            # Base class (checkpointing, AMP inference)
│   ├── ModelBuilder.py         # Model registry + optimizer/scheduler factories
│   ├── ResNet_VAE.py           # ResNet3d + VAE variant
│   ├── nnUnet.py               # Custom nnUNet with skip connections
│   ├── convnext.py             # 3D ConvNeXt
│   ├── mednext.py              # MedNeXt encoder-decoder
│   ├── survival_head.py        # Overall survival regression
│   └── layers.py               # Shared layer utilities
├── utils/
│   ├── configurator.py         # Experiment config (dataclass)
│   ├── transforms.py           # MONAI data transforms
│   ├── losses.py               # Dice, Focal, Boundary, Compound losses
│   ├── metrics.py              # BraTS metrics (Dice, HD95, sensitivity)
│   ├── postprocess.py          # Connected components, hierarchy enforcement
│   ├── inference.py            # Sliding window + ensemble prediction
│   ├── tta.py                  # Test-time augmentation
│   ├── uncertainty.py          # Monte Carlo dropout
│   ├── peft.py                 # LoRA, adapters, encoder freezing
│   ├── hub.py                  # HuggingFace Hub + MONAI bundle loading
│   ├── visualize.py            # Plotting utilities
│   └── expand_dim.py           # 2D→3D weight conversion
└── bundles/                    # MONAI bundle cache (auto-created)
```

## Tools & Frameworks

- **PyTorch** + **PyTorch Lightning** — training
- **MONAI** — medical imaging transforms, architectures, sliding window inference
- **Weights & Biases** — experiment tracking
- **HuggingFace Hub** — pre-trained model downloading
- **timm** — vision model components (DropPath, trunc_normal_)
