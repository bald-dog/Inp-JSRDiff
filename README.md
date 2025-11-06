# Inp-JSRDiff: Image Inpainting via Jointing Structure Restoration and End-to-end Reversible
Diffusion

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**High-Quality Image Inpainting System Based on Stable Diffusion**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Model Architecture](#model-architecture) • [Training](#training) • [Testing](#testing) • [Results](#results)

</div>

---

## 📖 Introduction

InpaintDM is an advanced image inpainting system based on diffusion models. It leverages the pre-trained Stable Diffusion v1.5 model, combined with innovative edge prior guidance and reversible backpropagation techniques, to achieve high-quality image inpainting. The system employs channel attention mechanisms and adaptive feature fusion strategies, demonstrating excellent performance in handling large missing regions.

### ✨ Key Features

- 🎯 **Edge Prior Guidance**: Integrates Canny edge detection as structural priors for clearer edge and structure restoration
- 🔄 **Reversible Backpropagation**: Uses RevBackProp technique to significantly reduce memory consumption and support deeper network structures
- 🎨 **Channel Attention Mechanism**: Adaptive feature weighting to optimize feature fusion
- ⚡ **Mixed Precision Training**: Utilizes Automatic Mixed Precision (AMP) to accelerate training
- 📊 **Comprehensive Evaluation Metrics**: Supports multiple evaluation metrics including PSNR, SSIM, LPIPS, FID
- 🔧 **Flexible Step Control**: Supports customizable diffusion steps to balance quality and speed
- 💾 **Resume Training**: Supports resuming training from checkpoints for training safety

---

## 🛠️ Installation

### System Requirements

- Python 3.8+
- CUDA 11.0+ (recommended for GPU acceleration)
- GPU with at least 16GB VRAM (for training)
- GPU with at least 8GB VRAM (for inference)

### Dependency Installation

```bash
# Clone the repository
git clone https://github.com/bald-dog/Inp-JSRDiff.git
cd InpaintDM

# Install PyTorch (choose according to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install diffusers transformers accelerate
pip install opencv-python scikit-image
pip install lpips scipy tqdm psutil
pip install tensorboard
```

### Mirror Acceleration (Optional, recommended for users in mainland China)

```bash
# Set Hugging Face mirror
export HF_ENDPOINT='https://hf-mirror.com'
```

---

## 🚀 Quick Start

### Train Model

```bash
bash train.sh
```

### Download Pre-trained Model

Pre-trained models can be downloaded from [BaiDu NetDisk](https://pan.baidu.com/s/1I6QLQFyQhnG1E1YHwrT1Vw). Extraction code: `fajm`

### Inference Script Example

```bash
bash test.sh
```

---

## 🏗️ Model Architecture

### Overall Architecture

The core architecture of InpaintDM is based on the following key components:

```
Input (Masked Image + Edge Prior)
    ↓
Feature Extraction (UNet Encoder)
    ↓
CADI Module (Feature Injection + Channel Attention)
    ↓
Diffusion Process (T-step Iterative Restoration)
    ↓
Feature Decoding (UNet Decoder)
    ↓
Output (Restored Image)
```
## 📊 Training Details

### Data Preparation

```
datasets/
├── paris_train_256/          # Training images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── paris_val_256/            # Validation images
│   └── ...
└── masks/                    # Mask files
    ├── mask_group/
    │   ├── 0-10/             # 0-10% mask ratio
    │   ├── 10-20/            # 10-20% mask ratio
    │   ├── 20-30/            # 20-30% mask ratio
    │   ├── 30-40/            # 30-40% mask ratio
    │   ├── 40-50/            # 40-50% mask ratio
    │   └── 50-60/            # 50-60% mask ratio
```

### Training Parameters

```bash
python train.py \
    --epoch 10                    # Number of training epochs
    --step_number 2               # Diffusion steps (T)
    --batch_size 4                # Batch size
    --patch_size 256              # Image patch size
    --learning_rate 1e-4          # Learning rate
    --save_interval 2             # Save interval (epochs)
    --train_sample_num 15000      # Number of training samples
    --preload_images 500          # Number of preloaded images
    --preload_masks 500           # Number of preloaded masks
    --num_workers 8               # Number of data loading workers
    --edge_mode full              # Edge prior mode
    --profile                     # Enable performance profiling
```

### Training Monitoring

The training process uses TensorBoard for real-time monitoring:

```bash
# Start TensorBoard
tensorboard --logdir=log/tensorboard

# Visit http://localhost:6006 to view training curves
```

**Visualization Contents:**
- Training loss curves
- Learning rate changes
- GPU memory usage
- Data loading time
- Training time analysis

### Performance Optimization

#### Data Loading Optimization
- **Preloading Mechanism**: Preload frequently used images and masks into memory
- **Multi-threaded Loading**: Parallel data loading using multiple workers
- **Non-blocking Transfer**: Asynchronous CPU-GPU data transfer

#### Training Optimization
- **Mixed Precision Training**: Uses FP16/FP32 mixed precision
- **Gradient Scaling**: Prevents gradient underflow
- **Gradient Accumulation**: Supports larger effective batch sizes
---
## 📝 Project Structure

```
InpaintDM/
├── model.py              # Main model definition (Net, Injector)
├── networks.py           # Network components (CBAM, Attention)
├── train.py              # Training script
├── test.py               # Testing script
├── utils.py              # Utility functions (PSNR, SSIM, etc.)
├── forward.py            # Forward propagation definition
├── backprop.py           # Reversible backpropagation
├── train.sh              # Training launch script
├── test.sh               # Testing launch script
├── weight/               # Model weights directory
│   ├── net_params_*.pkl  # Model parameters
│   └── checkpoint_*.pt   # Training checkpoints
├── log/                  # Training logs
│   ├── *.txt             # Text logs
│   └── tensorboard/      # TensorBoard logs
└── test_results/         # Test results
    └── epoch_*/          # Results for each epoch
```
---

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [IDM](https://github.com/Guaishou74851/IDM.git) - Invertible Diffusion Models for Compressed Sensing
- [Diffusers](https://github.com/huggingface/diffusers) - Diffusion models library
- [Edge-Connect](https://github.com/knazeri/edge-connect) - Edge-guided inpainting inspiration


---


## ⭐ Support Us

---

<div align="center">

**⭐ If this project helps you, please give us a Star! ⭐**

Made with ❤️ by [bald-dog](https://github.com/bald-dog)

</div>
