#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/4/25

@author: MaxML154
"""

import os
import json
import shutil
import argparse
from pycocotools.coco import COCO
import cv2
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Process NEU-DET dataset for few-shot object detection')
    parser.add_argument('--neu_det_path', required=True, help='Path to the extracted NEU-DET-METATRAIN dataset')
    parser.add_argument('--output_path', required=True, help='Path to store processed data')
    return parser.parse_args()

def convert_annotations_to_coco(neu_det_path, output_path):
    """Convert NEU-DET annotations to COCO format."""
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    annotations_dir = os.path.join(output_path, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Define base classes (training) and novel classes (testing)
    # From README_IFDD.md: base classes are inclusion, rolled-in scales and scratches
    # Novel classes are crazing, patches and pitted surface
    base_categories = [
        {"id": 1, "name": "inclusion"},
        {"id": 2, "name": "rolled-in_scales"},
        {"id": 3, "name": "scratches"}
    ]
    
    novel_categories = [
        {"id": 4, "name": "crazing"},
        {"id": 5, "name": "patches"},
        {"id": 6, "name": "pitted_surface"}
    ]
    
    # Process base annotations
    process_annotations(neu_det_path, output_path, 'train2017', base_categories, 'base_train.json')
    
    # Process novel annotations for validation
    process_annotations(neu_det_path, output_path, 'val2017', novel_categories, 'novel_val.json')
    
    # Create combined categories for future reference
    all_categories = base_categories + novel_categories
    with open(os.path.join(annotations_dir, 'all_categories.json'), 'w') as f:
        json.dump(all_categories, f)
    
    print("Dataset processing completed!")

def process_annotations(neu_det_path, output_path, split_name, categories, output_json):
    """Process annotations for a specific split."""
    
    # Base COCO structure
    coco_json = {
        "info": {
            "description": "NEU-DET adapted for few-shot object detection",
            "url": "https://ieeexplore.ieee.org/abstract/document/8709818",
            "version": "1.0",
            "year": 2021,
            "contributor": "MaxML154"
        },
        "licenses": [{"id": 1, "name": "Attribution-NonCommercial", "url": "http://creativecommons.org/licenses/by-nc/2.0/"}],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Load and process split data
    split_path = os.path.join(neu_det_path, split_name)
    
    # Map category names to IDs
    category_map = {cat["name"]: cat["id"] for cat in categories}
    
    # Process images and annotations
    image_id = 0
    annotation_id = 0
    
    for category_name, category_id in category_map.items():
        category_dir = os.path.join(split_path, category_name)
        
        if not os.path.exists(category_dir):
            continue
            
        for img_file in os.listdir(category_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Read image to get dimensions
            img_path = os.path.join(category_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            height, width, _ = img.shape
            
            # Create image entry
            image_entry = {
                "id": image_id,
                "file_name": f"{category_name}/{img_file}",
                "width": width,
                "height": height,
                "license": 1
            }
            coco_json["images"].append(image_entry)
            
            # For simplicity, we'll create an annotation covering the whole image
            # In reality, you should use actual bounding box annotations
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [0, 0, width, height],  # [x, y, width, height]
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            }
            coco_json["annotations"].append(annotation_entry)
            
            annotation_id += 1
            image_id += 1
    
    # Save annotations
    with open(os.path.join(output_path, 'annotations', output_json), 'w') as f:
        json.dump(coco_json, f)
    
    return image_id

def main():
    args = parse_args()
    convert_annotations_to_coco(args.neu_det_path, args.output_path)

if __name__ == "__main__":
    main() 