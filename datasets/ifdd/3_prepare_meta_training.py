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
from os.path import join, isdir
from os import mkdir, makedirs

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare NEU-DET-METATRAIN dataset for meta-training')
    parser.add_argument('--neu_det_path', required=True, help='Path to the extracted NEU-DET-METATRAIN dataset')
    parser.add_argument('--output_path', required=True, help='Path to store processed data')
    return parser.parse_args()

def crop_support(img, bbox):
    """Crop and resize support image with context."""
    image_shape = img.shape[:2]  # h, w
    data_height, data_width = image_shape
    
    img = img.transpose(2, 0, 1)

    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    
    width = x2 - x1
    height = y2 - y1
    context_pixel = 16
    
    new_x1 = 0
    new_y1 = 0
    new_x2 = width
    new_y2 = height
    target_size = (320, 320)
 
    if width >= height:
        crop_x1 = x1 - context_pixel
        crop_x2 = x2 + context_pixel
   
        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + context_pixel
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width
            
        short_size = height
        long_size = crop_x2 - crop_x1
        y_center = int((y2+y1) / 2)
        crop_y1 = int(y_center - (long_size / 2))
        crop_y2 = int(y_center + (long_size / 2))
        
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + np.ceil((long_size - short_size) / 2).astype(np.int32)
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height
        
        crop_short_size = crop_y2 - crop_y1
        crop_long_size = crop_x2 - crop_x1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype=np.uint8)
        delta = int((crop_long_size - crop_short_size) / 2)
        square_y1 = delta
        square_y2 = delta + crop_short_size

        new_y1 = new_y1 + delta
        new_y2 = new_y2 + delta
        
        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, square_y1:square_y2, :] = crop_box
    else:
        crop_y1 = y1 - context_pixel
        crop_y2 = y2 + context_pixel
   
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + context_pixel
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height
            
        short_size = width
        long_size = crop_y2 - crop_y1
        x_center = int((x2 + x1) / 2)
        crop_x1 = int(x_center - (long_size / 2))
        crop_x2 = int(x_center + (long_size / 2))

        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + np.ceil((long_size - short_size) / 2).astype(np.int32)
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width

        crop_short_size = crop_x2 - crop_x1
        crop_long_size = crop_y2 - crop_y1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype=np.uint8)
        delta = int((crop_long_size - crop_short_size) / 2)
        square_x1 = delta
        square_x2 = delta + crop_short_size

        new_x1 = new_x1 + delta
        new_x2 = new_x2 + delta
        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, :, square_x1:square_x2] = crop_box

    square = square.astype(np.float32, copy=False)
    square_scale = float(target_size[0]) / long_size
    square = square.transpose(1,2,0)
    square = cv2.resize(square, target_size, interpolation=cv2.INTER_LINEAR)
    square = square.astype(np.uint8)

    new_x1 = int(new_x1 * square_scale)
    new_y1 = int(new_y1 * square_scale)
    new_x2 = int(new_x2 * square_scale)
    new_y2 = int(new_y2 * square_scale)

    support_data = square
    support_box = np.array([new_x1, new_y1, new_x2, new_y2]).astype(np.float32)
    return support_data, support_box

def prepare_metatrain_dataset(neu_det_path, output_path):
    """Prepare NEU-DET-METATRAIN dataset for meta-training."""
    
    meta_train_path = os.path.join(output_path, 'NEU-DET-METATRAIN')
    os.makedirs(meta_train_path, exist_ok=True)
    
    # Create folder structure
    os.makedirs(os.path.join(meta_train_path, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(meta_train_path, 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(meta_train_path, 'val2017'), exist_ok=True)
    
    # Define base classes (training) and novel classes (testing)
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
    
    all_categories = base_categories + novel_categories
    
    # Create annotation structures
    train_annotations = {
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
        "categories": all_categories
    }
    
    val_annotations = {
        "info": train_annotations["info"],
        "licenses": train_annotations["licenses"],
        "images": [],
        "annotations": [],
        "categories": all_categories
    }
    
    # Process base classes (for train2017)
    image_id = 0
    annotation_id = 0
    
    # Process each base category
    for category in base_categories:
        category_name = category["name"]
        category_id = category["id"]
        
        category_path = os.path.join(neu_det_path, category_name)
        if not os.path.isdir(category_path):
            print(f"Warning: {category_path} directory not found!")
            continue
        
        # Create category directory in train2017
        cat_train_dir = os.path.join(meta_train_path, 'train2017', category_name)
        os.makedirs(cat_train_dir, exist_ok=True)
        
        # Process all images in this category
        for img_file in os.listdir(category_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Read image
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            height, width, _ = img.shape
            
            # Copy image to train2017
            dest_img_path = os.path.join(cat_train_dir, img_file)
            shutil.copy(img_path, dest_img_path)
            
            # Create image entry
            image_entry = {
                "id": image_id,
                "file_name": f"{category_name}/{img_file}",
                "width": width,
                "height": height,
                "license": 1
            }
            train_annotations["images"].append(image_entry)
            
            # Create annotation entry (using full image as bbox for simplicity)
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [0, 0, width, height],  # [x, y, width, height]
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            }
            train_annotations["annotations"].append(annotation_entry)
            
            annotation_id += 1
            image_id += 1
    
    # Process novel classes for validation (for val2017)
    val_image_id = 0
    val_annotation_id = 0
    
    # Process each novel category
    for category in novel_categories:
        category_name = category["name"]
        category_id = category["id"]
        
        category_path = os.path.join(neu_det_path, category_name)
        if not os.path.isdir(category_path):
            print(f"Warning: {category_path} directory not found!")
            continue
        
        # Create category directory in val2017
        cat_val_dir = os.path.join(meta_train_path, 'val2017', category_name)
        os.makedirs(cat_val_dir, exist_ok=True)
        
        # Process all images in this category
        for img_file in os.listdir(category_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Read image
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            height, width, _ = img.shape
            
            # Copy image to val2017
            dest_img_path = os.path.join(cat_val_dir, img_file)
            shutil.copy(img_path, dest_img_path)
            
            # Create image entry
            image_entry = {
                "id": val_image_id,
                "file_name": f"{category_name}/{img_file}",
                "width": width,
                "height": height,
                "license": 1
            }
            val_annotations["images"].append(image_entry)
            
            # Create annotation entry (using full image as bbox for simplicity)
            annotation_entry = {
                "id": val_annotation_id,
                "image_id": val_image_id,
                "category_id": category_id,
                "bbox": [0, 0, width, height],  # [x, y, width, height]
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            }
            val_annotations["annotations"].append(annotation_entry)
            
            val_annotation_id += 1
            val_image_id += 1
    
    # Save annotations
    with open(os.path.join(meta_train_path, 'annotations', 'train2017.json'), 'w') as f:
        json.dump(train_annotations, f)
    
    with open(os.path.join(meta_train_path, 'annotations', 'val2017.json'), 'w') as f:
        json.dump(val_annotations, f)
    
    # Generate support pool for meta-learning
    support_path = os.path.join(meta_train_path, 'support')
    os.makedirs(support_path, exist_ok=True)
    
    support_dict = {
        'support_box': [],
        'category_id': [],
        'image_id': [],
        'id': [],
        'file_path': []
    }
    
    # Generate support images for base classes
    for img_entry in train_annotations["images"]:
        img_id = img_entry["id"]
        img_path = os.path.join(meta_train_path, 'train2017', img_entry["file_name"])
        
        if not os.path.exists(img_path):
            continue
            
        im = cv2.imread(img_path)
        if im is None:
            continue
            
        anns = [ann for ann in train_annotations["annotations"] if ann["image_id"] == img_id]
        
        for item_id, ann in enumerate(anns):
            rect = ann['bbox']
            bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
            
            # Crop support image
            support_img, support_box = crop_support(im, bbox)
            
            # Save support image
            frame_crop_base_path = os.path.join(support_path, f"{img_id}")
            os.makedirs(frame_crop_base_path, exist_ok=True)
            
            file_path = os.path.join(frame_crop_base_path, f'{item_id:04d}.jpg')
            cv2.imwrite(file_path, support_img)
            
            # Add to support dict
            support_dict['support_box'].append(support_box.tolist())
            support_dict['category_id'].append(ann['category_id'])
            support_dict['image_id'].append(ann['image_id'])
            support_dict['id'].append(ann['id'])
            support_dict['file_path'].append(file_path)
    
    # Save support dictionary
    support_df = pd.DataFrame.from_dict(support_dict)
    support_df.to_pickle(os.path.join(meta_train_path, 'metatrain_support_df.pkl'))
    
    print("NEU-DET-METATRAIN preparation completed!")

def main():
    args = parse_args()
    prepare_metatrain_dataset(args.neu_det_path, args.output_path)

if __name__ == "__main__":
    main() 