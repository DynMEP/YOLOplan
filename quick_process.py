#!/usr/bin/env python3
# =============================================================================
# YOLOplan: Automated Symbol Detection for Technical Drawings
# =============================================================================
# Purpose: Complete YOLO11 implementation for automated symbol detection and 
#          counting in technical drawings (PDF, images, CAD) for MEP 
#          (Mechanical, Electrical, Plumbing) projects with netlist generation.
# Version: 1.1.0 (Fixed & Enhanced Edition)
# Author: Alfonso Davila - Electrical Engineer, Revit MEP Dynamo BIM Expert
# Contact: davila.alfonso@gmail.com - www.linkedin.com/in/alfonso-davila-3a121087
# Repository: https://github.com/DynMEP/YOLOplan
# License: MIT License (see LICENSE file in repository)
# YOLO11: AGPL-3.0 (Ultralytics) - Commercial license available from Ultralytics
# Created: October 2025
# Last Updated: October 8, 2025
# Compatibility: Python 3.8+, CUDA 11.8+ (optional), Poppler (for PDF support)
# Quick Start:
#   Single plan:  python quick_process.py my_plan.pdf
#   Multiple plans:   python quick_process.py plans_directory/
# =============================================================================

import os
import sys
from pathlib import Path
from YOLOplanDetector import YOLOplanDetector, PDFProcessor

def process_plan(pdf_path, output_dir, model='yolo11s.pt'):
    
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}")
    
    # Convert PDF to images
    temp_dir = os.path.join(output_dir, 'temp_images')
    processor = PDFProcessor()
    
    try:
        image_paths = processor.pdf_to_images(pdf_path, temp_dir, dpi=400)
        print(f"✓ Converted to {len(image_paths)} images")
        
        # Detect objects (will be generic without custom training)
        detector = YOLOplanDetector(model, conf_threshold=0.2)
        results = detector.process_batch(
            image_paths, 
            output_dir=output_dir,
            export_format='all'
        )
        
        print(f"✓ Processed {len(results)} images")
        print(f"✓ Results saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"1. Review annotated images in: {output_dir}/")
        print(f"2. Check detection counts in: {output_dir}/summary_*.csv")
        print(f"3. For better results, train custom model with your symbols")
        
        return results
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_process.py <pdf_file_or_directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if os.path.isfile(input_path):
        # Single PDF
        output = f"results_{Path(input_path).stem}"
        process_plan(input_path, output)
    
    elif os.path.isdir(input_path):
        # Multiple PDFs
        pdfs = list(Path(input_path).glob('*.pdf'))
        print(f"Found {len(pdfs)} PDF files")
        
        for pdf in pdfs:
            output = f"results_{pdf.stem}"
            process_plan(str(pdf), output)
    
    else:
        print(f"Error: {input_path} not found")