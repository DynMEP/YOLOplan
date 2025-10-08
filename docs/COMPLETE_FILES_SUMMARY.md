# YOLOplan Complete Files Summary

### Step 1: Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Step 2: Run Tests
```bash
pytest test_yoloplan.py -v
```

### Scenario: You have 5 electrical plans to process TODAY
# ===== STEP 1: Extract and Preview (5 min) =====
mkdir raw_plans processed_images

# Convert all PDFs to high-res images
for pdf in plans/*.pdf; do
    python YOLOplanDetector.py \
        --model yolo11n.pt \
        --source "$pdf" \
        --output "processed_images/$(basename $pdf .pdf)" \
        --dpi 400
done

# ===== STEP 2: Choose Your Path =====

# PATH A: Use existing model (if available)
# Download from Roboflow or use pretrained
python YOLOplanDetector.py \
    --model models/electrical_pretrained.pt \
    --source processed_images/ \
    --export all

# PATH B: Quick manual counting + AI assist
# Use YOLO to find ALL objects, then manually classify
python -c "
from YOLOplanDetector import YOLOplanDetector
import pandas as pd

detector = YOLOplanDetector('yolo11s.pt', conf_threshold=0.15)
results = detector.process_batch(
    ['processed_images/plan1_page_1.jpg'],
    export_format='csv'
)

# Now manually review annotated images and fix in Excel
print('Review results/annotated_images/ and edit results/details_*.csv')

# PATH C: Label 1 plan, train, use on other 4
# 1. Label symbols on plan_001 (30 min)
# 2. Quick train (30 min)
# 3. Run on plans 2-5 (5 min)