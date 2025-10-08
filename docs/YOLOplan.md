YOLOplan - YOLO11 Implementation
Complete implementation of YOLO11 for automated symbol detection and counting in technical drawings (PDF, images, CAD) for MEP (Mechanical, Electrical, Plumbing) projects. Updated with enhancements for adaptive preprocessing, hyperparameter optimization, synthetic data generation, and netlist generation.
🚀 Features

YOLO11 Integration: Latest YOLO architecture for state-of-the-art detection
Multi-format Support: PDF, JPG, PNG, BMP, TIFF
Batch Processing: Process multiple drawings simultaneously
Export Options: CSV, Excel, JSON formats with netlist support
Training Pipeline: Complete workflow with hyperparameter optimization
Dataset Tools: Adaptive preprocessing, synthetic data generation, and analysis
Annotated Outputs: Visual results with bounding boxes
Enhancements: Adaptive thresholding, real-time model fetching, netlist generation
Netlist Generation: Connectivity analysis for electrical schematics

📋 Requirements

Python 3.8 or higher
CUDA-capable GPU (recommended for training)
Poppler (for PDF processing)
Additional dependencies: optuna for hyperparameter optimization

🔧 Installation
1. Clone the repository
git clone https://github.com/DynMEP/YOLOplan.git
cd YOLOplan

2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Install Poppler (for PDF support)
Ubuntu/Debian:
sudo apt-get install poppler-utils

macOS:
brew install poppler

Windows:Download from: https://github.com/oschwartz10612/poppler-windows/releases
📁 Project Structure
YOLOplan/
├── ImagePreprocessor.py   # Preprocessing and dataset utilities
├── YOLOplanTrainer.py    # Training script with hyperparameter optimization
├── YOLOplanDetector.py   # Detection and netlist generation
├── requirements.txt      # Dependencies
├── datasets/
│   └── your_dataset/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
├── models/
│   └── your_model.pt
└── results/
    ├── annotated_images/
    ├── summary.csv
    ├── details.csv
    └── netlist.csv

🎯 Quick Start
Detection
Detect symbols in a single image:
python YOLOplanDetector.py --model models/yolo11n.pt --source path/to/image.jpg

Detect symbols in a PDF:
python YOLOplanDetector.py --model models/yolo11n.pt --source path/to/drawing.pdf --dpi 300

Batch process multiple images with netlist generation:
python YOLOplanDetector.py --model models/yolo11n.pt --source path/to/images/ --export all

Fetch latest model for a dataset (e.g., Skema Hybrid Method V2):
python YOLOplanDetector.py --model models/yolo11n.pt --source path/to/images/ --fetch-model skema

Training
1. Prepare your dataset
python YOLOplanTrainer.py setup --path datasets/electrical_symbols --classes outlet switch light fixture panel

This creates the following structure:
datasets/electrical_symbols/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml

2. Label your data
Use tools like:

Roboflow
LabelImg
CVAT

Export in YOLO format.
3. Train the model with optimization
python YOLOplanTrainer.py train \
    --data datasets/electrical_symbols/data.yaml \
    --model s \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --project runs/train \
    --name electrical_yolo11 \
    --optimize

Model sizes:

n (nano): Fastest, smallest
s (small): Balanced
m (medium): Good accuracy
l (large): High accuracy
x (extra large): Best accuracy

4. Fine-tune on new dataset
python YOLOplanTrainer.py train \
    --data datasets/new_symbols/data.yaml \
    --model s \
    --weights runs/train/electrical_yolo11/weights/best.pt \
    --epochs 50

5. Validate the model
python YOLOplanTrainer.py val \
    --weights runs/train/electrical_yolo11/weights/best.pt \
    --data datasets/electrical_symbols/data.yaml

6. Export the model
python YOLOplanTrainer.py export \
    --weights runs/train/electrical_yolo11/weights/best.pt \
    --format onnx \
    --imgsz 640

📊 Dataset Utilities
Generate Synthetic Data
Augment small datasets with synthetic schematics:
from ImagePreprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()
preprocessor.generate_synthetic_schematic(
    class_names=['outlet', 'switch', 'light', 'panel'],
    output_dir='datasets/synthetic',
    num_images=100
)

Analyze Dataset
from ImagePreprocessor import DatasetAnalyzer

analyzer = DatasetAnalyzer()
analyzer.analyze_dataset(
    labels_dir='datasets/electrical_symbols/labels/train',
    class_names=['outlet', 'switch', 'light', 'panel'],
    output_dir='analysis'
)

Output: class_distribution.png, bbox_statistics.png in analysis/.
Split Dataset
from ImagePreprocessor import DatasetSplitter

splitter = DatasetSplitter()
splitter.split_dataset(
    images_dir='raw_images',
    labels_dir='raw_labels',
    output_dir='datasets/electrical_symbols',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)

Preprocess Images (Parallel)
from ImagePreprocessor import batch_preprocess_images

batch_preprocess_images(
    input_dir='raw_images',
    output_dir='processed_images',
    enhance=True,
    resize=640
)

Convert COCO to YOLO
from ImagePreprocessor import LabelConverter

converter = LabelConverter()
converter.coco_to_yolo(
    coco_json='annotations.json',
    output_dir='labels',
    image_dir='images'
)

Auto-Generate data.yaml
from YOLOplanTrainer import YOLOplanTrainer

trainer = YOLOplanTrainer(data_yaml='dummy.yaml')
trainer.auto_generate_yaml(
    labels_dir='datasets/electrical_symbols/labels/train',
    dataset_path='datasets/electrical_symbols'
)

🎨 Advanced Detection Options
python YOLOplanDetector.py \
    --model models/electrical_symbols.pt \
    --source drawings/ \
    --output results/project_name \
    --conf 0.30 \
    --iou 0.45 \
    --export all \
    --dpi 300 \
    --fetch-model skema

Parameters:

--model: Path to YOLO11 weights
--source: Image, directory, or PDF file
--output: Output directory
--conf: Confidence threshold (0-1)
--iou: IOU threshold for NMS
--export: Export format (csv, excel, json, all)
--dpi: DPI for PDF conversion
--fetch-model: Fetch pretrained model for dataset (e.g., skema)

🔬 Training Tips
1. Data Quality

Use high-resolution images (300+ DPI)
Ensure consistent lighting and quality
Include diverse examples (rotations, scales, backgrounds)
Minimum 100-200 images per class
Balance class distribution with synthetic data

2. Training Configuration

Start with pretrained weights (--pretrained)
Use larger batch sizes if GPU memory allows
Enable early stopping (--patience)
Run hyperparameter optimization (--optimize)

3. Hyperparameter Tuning
python YOLOplanTrainer.py train \
    --data data.yaml \
    --model s \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --patience 50 \
    --optimize

4. Data Augmentation
Built-in augmentations:

HSV color jittering
Random flipping
Random scaling and translation
Mosaic augmentation
RandAugment

For advanced augmentation, add albumentations to requirements.txt.
📈 Results Interpretation
Output Files

Summary CSV (summary_TIMESTAMP.csv):

Image,Total_Symbols,Count_outlet,Count_switch,Count_light,Timestamp
drawing1.jpg,45,12,8,25,2025-01-15T10:30:00
drawing2.jpg,38,10,6,22,2025-01-15T10:30:05


Details CSV (details_TIMESTAMP.csv):

Image,Class,Confidence,X1,Y1,X2,Y2,Center_X,Center_Y,Area,ID
drawing1.jpg,outlet,0.95,100,200,150,250,125,225,2500,drawing1.jpg_0
drawing1.jpg,switch,0.89,300,400,340,440,320,420,1600,drawing1.jpg_1


Netlist CSV (netlist_TIMESTAMP.csv):

Image,Wire_ID,Connected_Symbols
drawing1.jpg,drawing1.jpg_10,drawing1.jpg_0,drawing1.jpg_1
drawing1.jpg,drawing1.jpg_11,drawing1.jpg_2,drawing1.jpg_3


JSON Export (results_TIMESTAMP.json): Full results including netlists.

Visual Outputs

Annotated images in results/annotated_images/ with bounding boxes and labels.

🛠️ Troubleshooting
CUDA Out of Memory
python YOLOplanTrainer.py train --data data.yaml --batch 8 --imgsz 416

Low Detection Accuracy

Run hyperparameter optimization: --optimize
Generate synthetic data for small datasets:preprocessor.generate_synthetic_schematic(class_names=['outlet', 'switch'], num_images=200)


Use larger model (e.g., YOLO11m)
Check analysis/class_distribution.png for imbalance

PDF Processing Issues

Verify Poppler installation
Increase --dpi 400 for better resolution
Use PyMuPDF fallback if pdf2image fails

Small Symbol Detection

Preprocess with enhance=True
Use higher --imgsz 1280 if GPU allows

💡 Use Cases
Electrical Takeoff with Netlist
python YOLOplanDetector.py \
    --model models/electrical.pt \
    --source electrical_plans/ \
    --output takeoff_results \
    --export all

Skema Hybrid Method V2 Workflow
# Download dataset
roboflow download --project skema-hybrid-method-v2 --format yolov8
# Preprocess
python ImagePreprocessor.py --input datasets/skema/images --output datasets/skema/processed
# Train
python YOLOplanTrainer.py train --data datasets/skema/data.yaml --model s --optimize
# Detect
python YOLOplanDetector.py --model runs/train/exp/weights/best.pt --source new_plan.pdf

CGHD Hand-Drawn Circuits
# Download dataset
kaggle datasets download -d johannesbayer/cghd1152
unzip cghd1152.zip -d datasets/cghd
# Convert COCO to YOLO
python -c "from ImagePreprocessor import LabelConverter; converter = LabelConverter(); converter.coco_to_yolo('datasets/cghd/annotations.json', 'datasets/cghd/labels', 'datasets/cghd/images')"
# Train
python YOLOplanTrainer.py train --data datasets/cghd/data.yaml --model m

🔄 Workflow Example
Complete End-to-End Workflow
Step 1: Collect and label data
python YOLOplanTrainer.py setup --path datasets/my_symbols --classes symbol1 symbol2 symbol3
# Label images using LabelImg or Roboflow

Step 2: Analyze dataset
from ImagePreprocessor import DatasetAnalyzer

DatasetAnalyzer.analyze_dataset(
    labels_dir='datasets/my_symbols/labels/train',
    class_names=['symbol1', 'symbol2', 'symbol3'],
    output_dir='analysis'
)

Step 3: Generate synthetic data (if needed)
from ImagePreprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()
preprocessor.generate_synthetic_schematic(
    class_names=['symbol1', 'symbol2', 'symbol3'],
    output_dir='datasets/my_symbols/synthetic'
)

Step 4: Train model
python YOLOplanTrainer.py train \
    --data datasets/my_symbols/data.yaml \
    --model s \
    --epochs 150 \
    --batch 16 \
    --project runs/train \
    --name my_symbols_v1 \
    --optimize

Step 5: Validate model
python YOLOplanTrainer.py val \
    --weights runs/train/my_symbols_v1/weights/best.pt \
    --data datasets/my_symbols/data.yaml

Step 6: Run inference
python YOLOplanDetector.py \
    --model runs/train/my_symbols_v1/weights/best.pt \
    --source test_drawings/ \
    --output production_results \
    --export all \
    --conf 0.30

Step 7: Review results

Check annotated images in production_results/
Review counts in summary_*.csv
Analyze detections in details_*.csv
Inspect connectivity in netlist_*.csv

📚 API Usage
Python API
from YOLOplanDetector import YOLOplanDetector

# Initialize detector
detector = YOLOplanDetector(
    model_path='models/electrical.pt',
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Single image detection
result = detector.detect_symbols(
    image_path='drawing.jpg',
    save_annotated=True,
    output_dir='results'
)
print(f"Found {result['total_detections']} symbols")
print(f"Class counts: {result['class_counts']}")
print(f"Netlist: {result['netlist']}")

# Batch processing
image_paths = ['drawing1.jpg', 'drawing2.jpg']
results = detector.process_batch(
    image_paths=image_paths,
    output_dir='batch_results',
    export_format='all'
)

# Export results
detector.export_results(
    results=results,
    output_dir='exports',
    format='excel'
)

PDF Processing
from YOLOplanDetector import PDFProcessor

# Convert PDF to images
processor = PDFProcessor()
image_paths = processor.pdf_to_images(
    pdf_path='technical_drawing.pdf',
    output_dir='temp_images',
    dpi=300
)

# Process with detector
detector = YOLOplanDetector(model_path='models/electrical.pt')
results = detector.process_batch(image_paths, output_dir='results')

Custom Preprocessing
from ImagePreprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Enhance drawing
enhanced = preprocessor.enhance_drawing(
    image_path='raw_drawing.jpg',
    output_path='enhanced_drawing.jpg',
    sharpen=True,
    denoise=True,
    contrast=True
)

# Remove background
clean = preprocessor.remove_background(
    image_path='drawing.jpg',
    output_path='clean_drawing.jpg',
    threshold=240
)

# Resize maintaining aspect ratio
resized = preprocessor.resize_maintain_aspect(
    image_path='large_drawing.jpg',
    target_size=640,
    output_path='resized_drawing.jpg'
)

🎓 Model Performance



Model
Size (MB)
Speed (ms)
mAP50
Parameters



YOLO11n
6.2
1.5
39.5
2.6M


YOLO11s
21.5
2.3
47.0
9.4M


YOLO11m
49.7
4.4
51.5
20.1M


YOLO11l
86.9
6.2
53.4
25.3M


YOLO11x
141.7
11.3
54.7
56.9M


Benchmarked on COCO dataset with 640x640 input
Recommendations

Fast inference: YOLO11n or YOLO11s
Balanced: YOLO11s or YOLO11m
High accuracy: YOLO11l or YOLO11x
Production: YOLO11s (best speed/accuracy trade-off)

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📝 Citation
If you use YOLOplan in your research or projects, please cite:
@software{yoloplan2025,
  author = {Alfonso Davila},
  title = {YOLOplan: Automated Symbol Detection for Technical Drawings},
  year = {2025},
  url = {https://github.com/DynMEP/YOLOplan},
  note = {YOLO11 implementation for MEP symbol detection}
}

Also cite YOLO11:
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
YOLO11 is licensed under AGPL-3.0 for open-source use. For commercial use, consider Ultralytics Enterprise License.
👤 Author
Alfonso Davila  

Email: davila.alfonso@gmail.com  
LinkedIn: alfonso-davila-3a121087  
GitHub: DynMEP  
YouTube: @DynMEP  
Website: dynmep.com (Coming Soon)

🙏 Acknowledgments

Ultralytics for YOLO11
The open-source computer vision community
All contributors and users of YOLOplan

📞 Support

📧 Email: davila.alfonso@gmail.com  
🐛 Issues: GitHub Issues  
💬 Discussions: GitHub Discussions


"Let's build smarter MEP workflows together! 🚧"
For more tools and automation scripts, visit github.com/DynMEP