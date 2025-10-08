# YOLOplan

**YOLOplan** automates symbol detection and counting in technical drawings (PDF, images, CAD) using YOLO11 object detection. Built for MEP professionals, it streamlines takeoff for electrical, HVAC, plumbing, and architectural projects—fast, accurate, and production-ready.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_yoloplan.py)

---

## 🚀 Features

### Core Capabilities
- ✅ **YOLO11 Integration** - State-of-the-art object detection
- ✅ **Multi-format Support** - PDF, JPG, PNG, BMP, TIFF
- ✅ **Batch Processing** - Process multiple drawings simultaneously
- ✅ **Custom Training** - Train on your specific symbols
- ✅ **Export Options** - CSV, Excel, JSON with netlist support
- ✅ **Production Ready** - Comprehensive error handling & logging

### Advanced Features (v1.1.0)
- 🆕 **Adaptive Preprocessing** - Smart enhancement based on image characteristics
- 🆕 **Netlist Generation** - Connectivity analysis for electrical schematics
- 🆕 **Hyperparameter Optimization** - Auto-tune training with Optuna
- 🆕 **Synthetic Data Generation** - Augment small datasets
- 🆕 **Parallel Processing** - 3x faster batch operations
- 🆕 **Comprehensive Testing** - 25+ unit tests for reliability

---

## 🌎 Applications

- **Electrical & MEP Engineering** - Automated symbol takeoff and counting
- **Construction Estimation** - Rapid quantity extraction from plans
- **BIM & Digital Modeling** - Integration with building information models
- **Facility Management** - Asset inventory from as-built drawings
- **QA/QC** - Verify plan completeness and symbol placement
- **Archival Digitization** - Extract data from legacy drawings

---

## ⚡ Quick Start

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/DynMEP/YOLOplan.git
cd YOLOplan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest test_yoloplan.py -v
```

### Using Pre-trained Models (Immediate Use)

```bash
# Option 1: Use YOLO11 pretrained (general objects)
python YOLOplanDetector.py \
    --model yolo11s.pt \
    --source your_plan.pdf \
    --output results \
    --dpi 400

# Option 2: Download electrical symbols model from Roboflow
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
project = rf.workspace().project('electrical-symbols')
dataset = project.version(1).download('yolov8')
"

# Use downloaded model
python YOLOplanDetector.py \
    --model electrical-symbols-1/weights/best.pt \
    --source your_plan.pdf \
    --export all
```

### Training Custom Model (2-3 hours)

```bash
# 1. Setup dataset structure
python YOLOplanTrainer.py setup \
    --path datasets/my_symbols \
    --classes outlet switch light panel

# 2. Label 20-50 images using Roboflow or LabelImg
#    - Upload to roboflow.com (easiest)
#    - Or use: pip install labelImg && labelImg

# 3. Train model with optimization
python YOLOplanTrainer.py train \
    --data datasets/my_symbols/data.yaml \
    --model s \
    --epochs 100 \
    --batch 16 \
    --optimize \
    --name my_model

# 4. Detect symbols in production
python YOLOplanDetector.py \
    --model runs/train/my_model/weights/best.pt \
    --source production_plans/ \
    --export all
```

---

## 📂 Project Structure

```text
YOLOplan/
├── YOLOplanDetector.py         # Detection and inference engine
├── YOLOplanTrainer.py          # Training pipeline
├── ImagePreprocessor.py        # Preprocessing utilities
├── requirements.txt            # Dependencies
├── test_yoloplan.py           # Comprehensive test suite
│
├── docs/                       # Documentation
│   ├── YOLOplan.md            # Detailed usage guide
│   ├── INSTALLATION_GUIDE.md
│   ├── QUICK_REFERENCE.md
│   └── COMPLETE_FILES_SUMMARY.md
│
├── datasets/                   # Your training datasets
│   └── electrical_symbols/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       └── data.yaml
│
├── models/                     # Model weights
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   └── best.pt
│
├── runs/                       # Training outputs
│   └── train/
│       └── exp/
│           └── weights/
│
└── results/                    # Detection outputs
    ├── annotated_images/
    ├── summary_*.csv
    ├── details_*.csv
    └── netlist_*.csv
```

---

## 🛠️ Requirements

### System Requirements
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for training)
- **GPU**: CUDA-capable NVIDIA GPU recommended (optional)

### Core Dependencies
```text
ultralytics>=8.3.0         # YOLO11
torch>=2.0.0               # PyTorch
opencv-python>=4.8.0       # Computer vision
pandas>=2.0.0              # Data handling
PyMuPDF>=1.23.0           # PDF processing
optuna>=3.0.0             # Hyperparameter optimization
```

See [requirements.txt](requirements.txt) for complete list.

---

## 📊 Performance

### Model Comparison

| Model | Size | Speed | mAP50 | Best For |
|-------|------|-------|-------|----------|
| YOLO11n | 6.2 MB | 1.5 ms | 39.5% | Fast inference |
| YOLO11s | 21.5 MB | 2.3 ms | 47.0% | **Production (recommended)** |
| YOLO11m | 49.7 MB | 4.4 ms | 51.5% | High accuracy |
| YOLO11l | 86.9 MB | 6.2 ms | 53.4% | Maximum accuracy |

### Processing Speed
- **PDF Conversion**: ~2-5 seconds per page (400 DPI)
- **Detection**: ~50-200 ms per image (depends on model)
- **Batch Processing**: 3x faster with parallel processing
- **Training**: ~1-3 hours for 100 epochs (with GPU)

---

## 📖 Documentation

### Quick Links
- 📘 [Complete Usage Guide](docs/YOLOplan.md) - Detailed documentation
- 🔧 [Installation Guide](docs/INSTALLATION_GUIDE.md) - Step-by-step setup
- ⚡ [Quick Reference](docs/QUICK_REFERENCE.md) - Common commands
- 🐛 [Fixes & Improvements](docs/FIXES_AND_IMPROVEMENTS.md) - Changelog

### Tutorials
- [Training Custom Models](docs/YOLOplan.md#training)
- [Batch Processing PDFs](docs/YOLOplan.md#pdf-processing)
- [Dataset Preparation](docs/YOLOplan.md#dataset-utilities)
- [Netlist Generation](docs/YOLOplan.md#netlist-generation)

---

## 🎓 Examples

### Example 1: Electrical Takeoff
```python
from YOLOplanDetector import YOLOplanDetector

# Initialize detector
detector = YOLOplanDetector('models/electrical.pt', conf_threshold=0.25)

# Process drawing
result = detector.detect_symbols('electrical_plan.pdf', save_annotated=True)

# Print results
print(f"Total symbols: {result['total_detections']}")
for symbol, count in result['class_counts'].items():
    print(f"  {symbol}: {count}")

# Export
detector.export_results([result], 'takeoff_results', format='excel')
```

### Example 2: Batch Processing
```bash
# Process entire project folder
python YOLOplanDetector.py \
    --model models/electrical.pt \
    --source "project_plans/*.pdf" \
    --output project_takeoff \
    --dpi 400 \
    --conf 0.3 \
    --export all
```

### Example 3: Generate Synthetic Training Data
```python
from ImagePreprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()
preprocessor.generate_synthetic_schematic(
    class_names=['outlet', 'switch', 'light', 'panel'],
    output_dir='datasets/synthetic',
    num_images=200,
    img_size=640
)
```

---

## 🧪 Testing

### Run Tests
```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest test_yoloplan.py -v

# Run with coverage
pytest test_yoloplan.py --cov=. --cov-report=html

# Run specific test
pytest test_yoloplan.py::TestImagePreprocessor -v
```

### Test Coverage
- ✅ Image preprocessing
- ✅ Dataset splitting & analysis
- ✅ Label format conversion
- ✅ Detection pipeline
- ✅ Training configuration
- ✅ End-to-end workflows

---

## 🎯 Roadmap

### v1.1.0 (Current - October 2025)
- ✅ Fixed critical bugs
- ✅ Comprehensive error handling
- ✅ Logging system
- ✅ Test suite
- ✅ Performance optimizations

### v1.2.0 (Planned)
- 🔜 Advanced wire tracing for netlists
- 🔜 Real-time detection (webcam/video)
- 🔜 Web dashboard for results
- 🔜 API endpoint generation
- 🔜 Database export support

### v2.0.0 (Future)
- 🔮 Multi-discipline symbol libraries
- 🔮 Cloud training infrastructure
- 🔮 BIM integration (Revit, AutoCAD)
- 🔮 Mobile app support

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new features
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/YOLOplan.git

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests before committing
pytest test_yoloplan.py -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**YOLO11** is licensed under AGPL-3.0 for open-source use. For commercial use, consider the [Ultralytics Enterprise License](https://ultralytics.com/license).

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com) for YOLO11
- The open-source computer vision community
- All contributors and users of YOLOplan

---

## 📞 Support & Contact

### Get Help
- 📧 **Email**: davila.alfonso@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/DynMEP/YOLOplan/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/DynMEP/YOLOplan/discussions)
- 📺 **YouTube**: [@DynMEP](https://youtube.com/@DynMEP)

### Connect
- 💼 **LinkedIn**: [alfonso-davila-3a121087](https://linkedin.com/in/alfonso-davila-3a121087)
- 🐙 **GitHub**: [DynMEP](https://github.com/DynMEP)
- 🌐 **Website**: [dynmep.com](http://dynmep.com) (Coming Soon)

---

## 📈 Citation

If you use YOLOplan in your research or projects, please cite:

```bibtex
@software{yoloplan2025,
  author = {Alfonso Davila},
  title = {YOLOplan: Automated Symbol Detection for Technical Drawings},
  year = {2025},
  url = {https://github.com/DynMEP/YOLOplan},
  version = {1.1.0},
  note = {YOLO11 implementation for MEP symbol detection}
}
```

---

## ⭐ Star History

If YOLOplan helps your work, please consider giving it a star! ⭐

---

## 💖 Support This Project

<p><a href="https://www.buymeacoffee.com/h1pot"> 
  <img align="left" src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="50" width="210" alt="h1pot" />
</a></p><br><br>

Your support helps maintain and improve YOLOplan!

---

<div align="center">

### _"Let's build smarter MEP workflows together! 🚧"_

**[Get Started](docs/INSTALLATION_GUIDE.md)** • 
**[Documentation](docs/YOLOplan.md)** • 
**[Examples](docs/QUICK_REFERENCE.md)** • 
**[Contributing](#-contributing)**

Made with ❤️ by [Alfonso Davila](https://github.com/DynMEP)

</div>
