# YOLOplan

**YOLOplan** automates symbol detection and counting in technical drawings (PDF, images, CAD) using advanced YOLO object detection. It streamlines takeoff for electrical, HVAC, and more—fast, accurate, and adaptable for noisy or complex plans.

---

## 🚀 Features

- Automatic detection and counting of technical symbols (e.g., electrical, HVAC, plumbing, architecture)
- Works with PDF, image, and CAD formats
- Robust against background noise and varied symbol size/rotation
- Supports custom model training for your own symbols
- Export results to CSV, Excel, JSON, or image with bounding boxes

## 🌎 Applications

- Electrical and MEP engineering
- Construction estimation and takeoff
- BIM and digital building modeling
- Facility management
- QA/QC for plans

## ⚡️ Quick Start

1. **Extract images from PDF plans**  
   Use `pdf2image` or `PyMuPDF` to convert your PDF to images.
2. **Detect symbols with YOLOplan**  
   Run detection on your images using the provided YOLO models or train your own.
3. **Export results**  
   Get counts and locations as CSV/Excel or annotated images.

## 📂 Project Structure

```text
📂 Project Structure
YOLOplan/
├── yolo_plan_core/             # Common utilities (PDF/image processing, inference)
├── yolo_plan_electric/         # Electrical symbols models/scripts
├── yolo_plan_hvac/             # (Future) HVAC models/scripts
├── datasets/                   # See datasets/ section below
├── notebooks/                  # See notebooks/ section below
├── demo_takeoff_electric.ipynb
└── README.md
```

## 🛠️ Requirements

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV, pdf2image, PyMuPDF

## 📖 License

MIT License

---

**Contributions welcome!**  
Join us to expand YOLOplan to more engineering disciplines and help automate technical drawing analysis.

