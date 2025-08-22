# E6Setup

One-line setup for CIFAR-10 training on fresh Ubuntu 24.04 systems.

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/SappieKonig/E6Setup/main/install.sh | bash
```

This command will:
- Install system dependencies (git, curl)
- Install uv Python environment manager
- Clone this repository
- Set up Python environment with PyTorch
- Download CIFAR-10 dataset
- Train a CNN model using all available GPUs

## What it does

The setup script automatically:
1. Installs minimal system dependencies
2. Installs [uv](https://docs.astral.sh/uv/) for Python environment management
3. Clones this repository to `~/E6Setup`
4. Creates a Python virtual environment with PyTorch and dependencies
5. Runs `main.py` which:
   - Downloads the CIFAR-10 dataset
   - Trains a CNN model with multi-GPU support
   - Saves the trained model as `cifar10_model.pth`

## Requirements

- Fresh Ubuntu 24.04 installation
- NVIDIA GPUs with CUDA support (optional, will fallback to CPU)
- Internet connection for downloading dependencies and dataset

## Manual Setup

If you prefer to set up manually:

```bash
git clone https://github.com/SappieKonig/E6Setup.git
cd E6Setup
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv run python main.py
```

## Files

- `install.sh` - Main installation script
- `main.py` - CIFAR-10 training script with multi-GPU support  
- `pyproject.toml` - Python dependencies (PyTorch, torchvision, numpy)
