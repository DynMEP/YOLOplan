# YOLOplan Installation Guide

Complete installation and setup guide for YOLOplan v1.1.0 (Fixed & Enhanced)

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space

### Recommended for Training
- **GPU**: CUDA-capable NVIDIA GPU (6GB+ VRAM)
- **RAM**: 16GB+
- **CPU**: Multi-core processor (4+ cores)

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Clone from GitHub
```bash
# Clone the repository
git clone https://github.com/DynMEP/YOLOplan.git
cd YOLOplan

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python -c "from ultralytics import YOLO; print('✓ Installation successful!')"
```

### Option 2: Manual Setup
```bash
# Create project directory
mkdir YOLOplan
cd YOLOplan

# Download fixed files (copy all provided fixed files)
# - YOLOplanDetector.py
# - YOLOplanTrainer.py
# - ImagePreprocessor.py
# - requirements.txt
# - test_yoloplan.py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Detailed Installation

### Step 1: Python Environment Setup

#### Check Python Version
```bash
python --version  # Should be 3.8 or higher
```

#### Install Python if Needed
- **Ubuntu/Debian**:
  ```bash
  sudo apt update
  sudo apt install python3.9 python3.9-venv python3-pip
  ```

- **macOS** (using Homebrew):
  ```bash
  brew install python@3.9
  ```

- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### Step 2: Virtual Environment

```bash
# Create virtual environment
python -m venv yoloplan_env

# Activate it
# Linux/Mac:
source yoloplan_env/bin/activate

# Windows (Command Prompt):
yoloplan_env\Scripts\activate.bat

# Windows (PowerShell):
yoloplan_env\Scripts\Activate.ps1
```

### Step 3: Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision

# Install Ultralytics YOLO
pip install ultralytics>=8.3.0
```

### Step 4: Install Image Processing Libraries

```bash
# OpenCV
pip install opencv-python opencv-contrib-python

# PIL/Pillow
pip install Pillow

# PDF processing
pip install PyMuPDF pdf2image
```

### Step 5: Install Data Science Libraries

```bash
# Core data libraries
pip install numpy pandas

# Visualization
pip install matplotlib seaborn

# Machine learning utilities
pip install scikit-learn

# Excel support
pip install openpyxl

# Progress bars
pip install tqdm

# Hyperparameter optimization
pip install optuna
```

### Step 6: Install Poppler (for PDF support)

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

#### macOS
```bash
brew install poppler
```

#### Windows
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\Program Files\poppler`
3. Add to PATH:
   ```
   C:\Program Files\poppler\Library\bin
   ```

---

## 🧪 Verify Installation

### Quick Test
```bash
python -c "
import torch
import cv2
import pandas as pd
from ultralytics import YOLO
print('✓ All core libraries imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Run Test Suite
```bash
# Install pytest
pip install pytest pytest-cov

# Run tests
pytest test_yoloplan.py -v

# Expected output: All tests should PASS
```

### Test Detection Script
```bash
# Download a sample model (nano version, smallest)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Create a test image
python -c "
import cv2
import numpy as np
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
cv2.imwrite('test_image.jpg', img)
print('✓ Test image created')
"

# Test detection (will create model if not exists)
python YOLOplanDetector.py \
    --model yolo11n.pt \
    --source test_image.jpg \
    --output test_results

# Check results
ls test_results/
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Not Available
**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. OpenCV Import Error
**Symptom**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution (Ubuntu/Debian)**:
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

#### 3. PDF Processing Failed
**Symptom**: `pdf2image.exceptions.PDFInfoNotInstalledError`

**Solution**:
```bash
# Verify Poppler installation
which pdftoppm  # Linux/Mac
where pdftoppm  # Windows

# If not found, reinstall Poppler (see Step 6)
```

#### 4. Out of Memory During Training
**Symptom**: `CUDA out of memory` or system freeze

**Solutions**:
```bash
# Reduce batch size
python YOLOplanTrainer.py train --batch 8

# Reduce image size
python YOLOplanTrainer.py train --imgsz 416

# Use smaller model
python YOLOplanTrainer.py train --model n  # nano
```

#### 5. Permission Denied on Windows
**Symptom**: `PermissionError` when creating files

**Solution**:
Run terminal as Administrator or change output directory:
```bash
python YOLOplanDetector.py --output %USERPROFILE%\YOLOplan\results
```

#### 6. Slow Processing
**Symptom**: Very slow image processing

**Solutions**:
```python
# Enable parallel processing
from ImagePreprocessor import batch_preprocess_images
batch_preprocess_images(
    'input', 'output', 
    enhance=True, 
    max_workers=4  # Adjust based on CPU cores
)
```

---

## 🎓 Post-Installation Setup

### 1. Download Pretrained Models

```bash
# Create models directory
mkdir models

# Download YOLO11 models (run Python script)
python -c "
from ultralytics import YOLO
for size in ['n', 's', 'm', 'l']:
    model = YOLO(f'yolo11{size}.pt')
    print(f'✓ Downloaded yolo11{size}.pt')
"
```

### 2. Setup Dataset Structure

```bash
# Create dataset directories
python YOLOplanTrainer.py setup \
    --path datasets/electrical_symbols \
    --classes outlet switch light fixture panel
```

### 3. Configure Logging (Optional)

Create `logging_config.yaml`:
```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: default
    level: INFO
  file:
    class: logging.FileHandler
    filename: yoloplan.log
    formatter: default
    level: DEBUG
root:
  level: INFO
  handlers: [console, file]
```

---

## 🐳 Docker Installation (Advanced)

### Dockerfile
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run tests
RUN pytest test_yoloplan.py

# Default command
CMD ["python3", "YOLOplanDetector.py", "--help"]
```

### Build and Run
```bash
# Build image
docker build -t yoloplan:latest .

# Run container
docker run --gpus all -v $(pwd)/data:/app/data yoloplan:latest \
    python3 YOLOplanDetector.py \
    --model models/yolo11s.pt \
    --source /app/data/drawings/ \
    --output /app/data/results
```

---

## 📱 Platform-Specific Notes

### Windows Specifics
- Use `\` for paths or raw strings: `r"C:\path\to\file"`
- Some features may require Windows Subsystem for Linux (WSL2)
- Visual Studio Build Tools required for some packages

### macOS Specifics  
- M1/M2 chips: Use `pip install torch torchvision` (MPS support)
- May need to allow apps in Security & Privacy settings

### Linux Specifics
- May need `libgomp` for OpenMP support: `sudo apt install libgomp1`
- Permissions: Use `chmod +x` on scripts if needed

---

## 🔄 Updating YOLOplan

### Update to Latest Version
```bash
# Pull latest changes
git pull origin main

# Upgrade dependencies
pip install --upgrade -r requirements.txt

# Run tests to verify
pytest test_yoloplan.py -v
```

### Rollback if Issues
```bash
git checkout <previous-commit>
pip install -r requirements.txt
```

---

## 📊 Performance Optimization

### CPU Optimization
```bash
# Set OpenCV threads
export OPENCV_NUM_THREADS=4

# Set PyTorch threads
export OMP_NUM_THREADS=4
```

### GPU Optimization
```bash
# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Enable TensorFloat-32
export NVIDIA_TF32_OVERRIDE=1
```

### Memory Optimization
```python
# In your training script
import torch
torch.backends.cudnn.benchmark = True  # Faster but more memory
# or
torch.backends.cudnn.benchmark = False  # Slower but less memory
```

---

## 🆘 Getting Help

### Resources
- **Documentation**: [GitHub Wiki](https://github.com/DynMEP/YOLOplan/wiki)
- **Issues**: [GitHub Issues](https://github.com/DynMEP/YOLOplan/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DynMEP/YOLOplan/discussions)
- **Email**: davila.alfonso@gmail.com

### Before Asking for Help
1. Check this installation guide
2. Review FIXES_AND_IMPROVEMENTS.md
3. Run tests: `pytest test_yoloplan.py -v`
4. Check log files for errors
5. Search existing GitHub issues

### Reporting Issues
Include:
- OS and Python version
- Full error traceback
- Steps to reproduce
- Expected vs actual behavior

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All requirements installed (`pip install -r requirements.txt`)
- [ ] Poppler installed (for PDF support)
- [ ] CUDA available (for GPU training)
- [ ] Tests passing (`pytest test_yoloplan.py`)
- [ ] Sample detection successful
- [ ] Models directory created
- [ ] Dataset structure setup

---

**You're all set! Start detecting symbols in your technical drawings! 🎉**

Next steps: Check out the main README.md for usage examples and workflows.