# YOLOplan Quick Reference Guide

Fast reference for common commands and workflows.

---

## 🚀 Common Commands

### Detection

```bash
# Single image
python YOLOplanDetector.py --model models/yolo11s.pt --source image.jpg

# Multiple images
python YOLOplanDetector.py --model models/yolo11s.pt --source images/

# PDF drawing
python YOLOplanDetector.py --model models/yolo11s.pt --source drawing.pdf --dpi 400

# With all exports
python YOLOplanDetector.py --model models/yolo11s.pt --source images/ --export all

# Custom confidence
python YOLOplanDetector.py --model models/yolo11s.pt --source image.jpg --conf 0.35
```

### Training

```bash
# Basic training
python YOLOplanTrainer.py train --data data.yaml --model s --epochs 100

# With optimization
python YOLOplanTrainer.py train --data data.yaml --model s --optimize

# Fine-tuning
python YOLOplanTrainer.py train --data data.yaml --weights best.pt --epochs 50

# Small batch (low memory)
python YOLOplanTrainer.py train --data data.yaml --batch 8 --imgsz 416
```

### Validation

```bash
python YOLOplanTrainer.py val --weights best.pt --data data.yaml
```

### Export

```bash
# ONNX
python YOLOplanTrainer.py export --weights best.pt --format onnx

# TorchScript
python YOLOplanTrainer.py export --weights best.pt --format torchscript
```

### Dataset Setup

```bash
# Create structure
python YOLOplanTrainer.py setup --path datasets/my_data --classes cls1 cls2 cls3
```

---

## 📝 Python API Quick Reference

### Detection

```python
from YOLOplanDetector import YOLOplanDetector

# Initialize
detector = YOLOplanDetector('models/yolo11s.pt', conf_threshold=0.25)

# Single detection
result = detector.detect_symbols('image.jpg', save_annotated=True)
print(f"Found {result['total_detections']} symbols")

# Batch detection
results = detector.process_batch(['img1.jpg', 'img2.jpg'], export_format='all')
```

### Training

```python
from YOLOplanTrainer import YOLOplanTrainer

# Initialize
trainer = YOLOplanTrainer('data.yaml', model_size='s')

# Train
results = trainer.train(epochs=100, batch=16, optimize=True)

# Validate
val_results = trainer.validate(weights='best.pt')

# Export
trainer.export_model('best.pt', format='onnx')
```

### Preprocessing

```python
from ImagePreprocessor import ImagePreprocessor, batch_preprocess_images

# Single image enhancement
preprocessor = ImagePreprocessor()
enhanced = preprocessor.enhance_drawing('input.jpg', 'output.jpg')

# Batch processing
batch_preprocess_images('input_dir/', 'output_dir/', enhance=True, resize=640)

# Synthetic data
preprocessor.generate_synthetic_schematic(
    class_names=['outlet', 'switch'],
    output_dir='synthetic/',
    num_images=100
)
```

### Dataset Operations

```python
from ImagePreprocessor import DatasetSplitter, DatasetAnalyzer

# Split dataset
splitter = DatasetSplitter()
splitter.split_dataset('images/', 'labels/', 'output/', train_ratio=0.7)

# Analyze dataset
analyzer = DatasetAnalyzer()
analyzer.analyze_dataset('labels/', ['class1', 'class2'], 'analysis/')
```

### Label Conversion

```python
from ImagePreprocessor import LabelConverter

converter = LabelConverter()
converter.coco_to_yolo('annotations.json', 'yolo_labels/', 'images/')
```

---

## 🎯 Configuration Templates

### data.yaml Template
```yaml
path: datasets/my_dataset
train: images/train
val: images/val
test: images/test
nc: 3
names: ['class1', 'class2', 'class3']
```

### Hyperparameter Tuning
```python
train_args = {
    'lr0': 0.01,           # Initial learning rate
    'lrf': 0.01,           # Final learning rate
    'momentum': 0.937,     # Momentum
    'weight_decay': 0.0005,# Weight decay
    'warmup_epochs': 3,    # Warmup epochs
    'box': 7.5,            # Box loss weight
    'cls': 0.5,            # Class loss weight
    'hsv_h': 0.015,        # HSV hue augmentation
    'hsv_s': 0.7,          # HSV saturation
    'hsv_v': 0.4,          # HSV value
    'mosaic': 1.0,         # Mosaic augmentation
    'mixup': 0.0,          # Mixup augmentation
}
```

---

## 🔍 Debugging Commands

### Check Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

### Verbose Logging
```bash
python YOLOplanDetector.py --source image.jpg --model yolo11s.pt --verbose
```

### Test Individual Components
```python
# Test detector complexity estimation
from YOLOplanDetector import YOLOplanDetector
import cv2
img = cv2.imread('test.jpg')
detector = YOLOplanDetector('yolo11n.pt')
complexity = detector.estimate_image_complexity(img)
print(f"Image complexity: {complexity}")

# Test preprocessor
from ImagePreprocessor import ImagePreprocessor
preprocessor = ImagePreprocessor()
noise = preprocessor.estimate_noise(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
print(f"Noise level: {noise}")
```

---

## 📊 Output Files Reference

### Detection Outputs

**summary_TIMESTAMP.csv**
```csv
Image,Total_Symbols,Count_outlet,Count_switch,Timestamp
drawing1.jpg,45,12,8,2025-10-08T10:30:00
```

**details_TIMESTAMP.csv**
```csv
Image,Class,Confidence,X1,Y1,X2,Y2,Center_X,Center_Y,Area,ID
drawing1.jpg,outlet,0.95,100,200,150,250,125,225,2500,drawing1.jpg_0
```

**netlist_TIMESTAMP.csv**
```csv
Image,Wire_ID,Connected_Symbols
drawing1.jpg,wire_0,"outlet_1,switch_2"
```

### Training Outputs

**Directory Structure**
```
runs/train/exp/
├── weights/
│   ├── best.pt        # Best model
│   └── last.pt        # Last checkpoint
├── results.png        # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
└── R_curve.png
```

---

## 🎨 Common Workflows

### Workflow 1: Quick Detection
```bash
# 1. Download model (automatic on first run)
# 2. Detect
python YOLOplanDetector.py --model yolo11s.pt --source drawings/ --export all
# 3. Check results/ directory
```

### Workflow 2: Custom Training
```bash
# 1. Setup dataset
python YOLOplanTrainer.py setup --path datasets/custom --classes c1 c2 c3

# 2. Label images (use LabelImg, Roboflow, etc.)

# 3. Analyze
python -c "
from ImagePreprocessor import DatasetAnalyzer
DatasetAnalyzer.analyze_dataset('datasets/custom/labels/train', ['c1','c2','c3'], 'analysis')
"

# 4. Train
python YOLOplanTrainer.py train --data datasets/custom/data.yaml --model s --optimize

# 5. Validate
python YOLOplanTrainer.py val --weights runs/train/exp/weights/best.pt --data datasets/custom/data.yaml

# 6. Use
python YOLOplanDetector.py --model runs/train/exp/weights/best.pt --source test/
```

### Workflow 3: Data Augmentation
```python
from ImagePreprocessor import ImagePreprocessor, DatasetSplitter

# 1. Enhance existing images
preprocessor = ImagePreprocessor()
preprocessor.batch_preprocess_parallel('raw/', 'enhanced/', enhance=True)

# 2. Generate synthetic data
preprocessor.generate_synthetic_schematic(['c1', 'c2'], 'synthetic/', 200)

# 3. Combine and split
# (manually combine directories, then:)
splitter = DatasetSplitter()
splitter.split_dataset('all_images/', 'all_labels/', 'dataset/', 0.7, 0.2, 0.1)
```

---

## 🔧 Performance Tuning

### Speed Optimization
```bash
# Faster inference (lower accuracy)
--model yolo11n.pt --conf 0.3 --iou 0.5

# Parallel preprocessing
python -c "batch_preprocess_images('in/', 'out/', max_workers=8)"
```

### Accuracy Optimization
```bash
# Better accuracy (slower)
--model yolo11x.pt --conf 0.2 --iou 0.4

# Higher resolution
--imgsz 1280

# Training with optimization
--optimize --epochs 200
```

### Memory Optimization
```bash
# Low memory training
--batch 4 --imgsz 416 --model n

# Low memory inference
--model yolo11n.pt
```

---

## 📈 Metrics Interpretation

### Training Metrics
- **mAP50**: Mean Average Precision at 50% IoU (higher is better)
- **mAP50-95**: mAP averaged over IoU thresholds 50-95% (more strict)
- **Precision**: TP / (TP + FP) - how many detections are correct
- **Recall**: TP / (TP + FN) - how many objects are found
- **Box Loss**: Lower is better
- **Class Loss**: Lower is better

### Good Training Signs
- ✅ Losses decreasing smoothly
- ✅ mAP50 > 0.5 (50%+)
- ✅ No overfitting (val loss similar to train loss)

### Bad Training Signs
- ❌ Loss not decreasing
- ❌ Val loss >> train loss (overfitting)
- ❌ mAP50 < 0.3 (30%)

---

## 🚨 Common Error Solutions

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `--batch` size or `--imgsz` |
| `FileNotFoundError` | Check paths, use absolute paths |
| `No images found` | Check file extensions, permissions |
| `Model not found` | Download model or check path |
| `PDF conversion failed` | Install Poppler |
| `Import error` | `pip install -r requirements.txt` |
| `Slow processing` | Use `--model n` or enable GPU |

---

## 💡 Tips & Tricks

### Tip 1: Start Small
```bash
# Test on small dataset first
--epochs 10 --batch 8
```

### Tip 2: Use Pretrained Models
```bash
# Always start with pretrained weights
--pretrained  # (default in train command)
```

### Tip 3: Monitor Training
```bash
# Use TensorBoard (optional)
pip install tensorboard
tensorboard --logdir runs/train
```

### Tip 4: Batch Processing
```python
# Process multiple PDFs
import glob
pdfs = glob.glob('drawings/*.pdf')
for pdf in pdfs:
    # Process each PDF
    pass
```

### Tip 5: Save Configurations
```bash
# Save successful training command
python YOLOplanTrainer.py train \
    --data data.yaml \
    --model s \
    --epochs 100 \
    --batch 16 \
    > training_config.txt 2>&1
```

---

## 📚 Additional Resources

- **YOLO Documentation**: https://docs.ultralytics.com
- **GitHub**: https://github.com/DynMEP/YOLOplan
- **Roboflow**: https://roboflow.com (for labeling)
- **LabelImg**: https://github.com/heartexlabs/labelImg
- **CVAT**: https://cvat.ai

---

**Save this guide for quick reference during your projects!** 📌