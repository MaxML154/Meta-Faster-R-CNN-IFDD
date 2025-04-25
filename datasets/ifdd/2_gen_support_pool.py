#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/4/25

@author: MaxML154
"""

from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import shutil
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate support pool for NEU-DET')
    parser.add_argument('--data_path', required=True, help='Path to the processed data')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of shots (5, 10, or 30)')
    parser.add_argument('--num_sets', type=int, default=100, help='Number of support sets to generate')
    return parser.parse_args()

def vis_image(im, bboxs, im_name):
    dpi = 300
    fig, ax = plt.subplots() 
    ax.imshow(im, aspect='equal') 
    plt.axis('off') 
    height, width, channels = im.shape 
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    
    # Show box (off by default, box_alpha=0.0)
    for bbox in bboxs:
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='r',
                          linewidth=0.5, alpha=1))
    output_name = os.path.basename(im_name)
    plt.savefig(im_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close('all')

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
        new_y1 = new_y1 + math.ceil((long_size - short_size) / 2)
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
        new_x1 = new_x1 + math.ceil((long_size - short_size) / 2)
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

def generate_fewshot_sets(data_path, k_shot, num_sets):
    """Generate k-shot support sets for testing."""
    
    # Output path for k-shot dataset
    kshot_output_path = os.path.join(data_path, f'{k_shot}shot1-{num_sets}')
    os.makedirs(kshot_output_path, exist_ok=True)
    
    # Load categories
    with open(os.path.join(data_path, 'annotations', 'all_categories.json'), 'r') as f:
        all_categories = json.load(f)
    
    # Separate base and novel categories
    base_categories = all_categories[:3]  # inclusion, rolled-in scales, scratches
    novel_categories = all_categories[3:]  # crazing, patches, pitted surface
    
    # Load base annotations
    base_ann_file = os.path.join(data_path, 'annotations', 'base_train.json')
    base_coco = COCO(base_ann_file)
    
    # Load novel annotations
    novel_ann_file = os.path.join(data_path, 'annotations', 'novel_val.json')
    novel_coco = COCO(novel_ann_file)
    
    # Prepare query set (validation set stays the same for all k-shot sets)
    val_path = os.path.join(data_path, 'val2017')
    
    # Generate k-shot sets
    for i in range(1, num_sets + 1):
        kshot_dir = os.path.join(kshot_output_path, f'{k_shot}shot{i}')
        os.makedirs(kshot_dir, exist_ok=True)
        
        # Create folder structure
        os.makedirs(os.path.join(kshot_dir, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(kshot_dir, 'train2017'), exist_ok=True)
        os.makedirs(os.path.join(kshot_dir, 'val2017'), exist_ok=True)
        
        # Copy validation set (query set)
        for img_id in novel_coco.imgs:
            img = novel_coco.loadImgs(img_id)[0]
            src_img_path = os.path.join(val_path, img['file_name'])
            dest_img_path = os.path.join(kshot_dir, 'val2017', os.path.basename(img['file_name']))
            
            # Create category directories if needed
            category_dir = os.path.dirname(dest_img_path)
            os.makedirs(category_dir, exist_ok=True)
            
            # Copy image file
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dest_img_path)
        
        # Create empty support set annotation structure
        support_annotations = {
            "info": novel_coco.dataset['info'],
            "licenses": novel_coco.dataset['licenses'],
            "images": [],
            "annotations": [],
            "categories": all_categories
        }
        
        # Sample k-shot support examples for each novel category
        train_imgs = []
        train_anns = []
        
        for cat in novel_categories:
            cat_id = cat['id']
            cat_name = cat['name']
            
            # Get all images for this category
            img_ids = novel_coco.getImgIds(catIds=[cat_id])
            
            # Randomly select k images
            selected_img_ids = random.sample(img_ids, min(k_shot, len(img_ids)))
            
            for img_id in selected_img_ids:
                img = novel_coco.loadImgs(img_id)[0]
                anns = novel_coco.loadAnns(novel_coco.getAnnIds(imgIds=img_id, catIds=[cat_id]))
                
                # Add to support set
                train_imgs.append(img)
                train_anns.extend(anns)
                
                # Copy image to train2017
                src_img_path = os.path.join(val_path, img['file_name'])
                
                # Create category subdirectory in train2017
                cat_dir = os.path.join(kshot_dir, 'train2017', cat_name)
                os.makedirs(cat_dir, exist_ok=True)
                
                dest_img_path = os.path.join(cat_dir, os.path.basename(img['file_name']))
                
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dest_img_path)
        
        # Update support annotations
        support_annotations["images"] = train_imgs
        support_annotations["annotations"] = train_anns
        
        # Save train2017 annotations
        with open(os.path.join(kshot_dir, 'annotations', 'train2017.json'), 'w') as f:
            json.dump(support_annotations, f)
        
        # Save val2017 annotations (same as novel annotations)
        with open(os.path.join(kshot_dir, 'annotations', 'val2017.json'), 'w') as f:
            json.dump(novel_coco.dataset, f)
        
        # Generate support pool for meta-learning
        support_dir = os.path.join(kshot_dir, 'support')
        os.makedirs(support_dir, exist_ok=True)
        
        support_dict = {
            'support_box': [],
            'category_id': [],
            'image_id': [],
            'id': [],
            'file_path': []
        }
        
        for img in train_imgs:
            img_id = img['id']
            img_path = os.path.join(kshot_dir, 'train2017', img['file_name'])
            
            if not os.path.exists(img_path):
                continue
                
            im = cv2.imread(img_path)
            if im is None:
                continue
                
            anns = [ann for ann in train_anns if ann['image_id'] == img_id]
            
            for item_id, ann in enumerate(anns):
                rect = ann['bbox']
                bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
                
                # Crop support image
                support_img, support_box = crop_support(im, bbox)
                
                # Save support image
                frame_crop_base_path = os.path.join(support_dir, f"{img_id}")
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
        support_df.to_pickle(os.path.join(kshot_dir, f'{k_shot}shot{i}_support_df.pkl'))
        
        # Create train/val/test text files (for reference)
        with open(os.path.join(kshot_dir, 'trainval.txt'), 'w') as f:
            for img in train_imgs:
                f.write(f"{img['file_name']}\n")
                
        with open(os.path.join(kshot_dir, 'test.txt'), 'w') as f:
            for img_id in novel_coco.imgs:
                img = novel_coco.loadImgs(img_id)[0]
                f.write(f"{img['file_name']}\n")
        
        print(f"Generated {k_shot}-shot support set {i}/{num_sets}")

def main():
    args = parse_args()
    generate_fewshot_sets(args.data_path, args.k_shot, args.num_sets)

if __name__ == "__main__":
    main() 