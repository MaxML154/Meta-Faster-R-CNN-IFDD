# Meta Faster R-CNN with IFDD Dataset

This is an adaptation of Meta Faster R-CNN to work with the IFDD (Few-shot NEU-DET, FS-ND) dataset for few-shot steel surface defect detection.

***Note that the current modifications are only effective on my device, and the configuration may differ on other devices. You may need to reconfigure. Additionally, due to differences in device performance, the final results may not be the same.***

## Table of Contents

- [IFDD Dataset](#ifdd-dataset)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation Process](#training-and-evaluation-process)
  - [1. Meta-training](#1-meta-training)
  - [2. Training Base Classes Detection Head](#2-training-base-classes-detection-head)
  - [3. Few-shot Fine-tuning](#3-few-shot-fine-tuning)
- [Evaluation](#evaluation)
- [Configurations](#configurations)
- [Credits](#credits)
- [About Meta Faster R-CNN](#about-meta-faster-r-cnn)
  - [Key Features of Meta Faster R-CNN](#key-features-of-meta-faster-r-cnn)
  - [Adaptation for IFDD Dataset](#adaptation-for-ifdd-dataset)

## IFDD Dataset

Few-shot NEU-DET (FS-ND) is a dataset for few-shot steel surface defect detection, reconstructed from the famous [NEU-DET](https://ieeexplore.ieee.org/abstract/document/8709818) dataset.

The dataset includes 6 classes of steel surface defects:
- Base classes (used in meta-training): 
  - inclusion
  - rolled-in scales
  - scratches
- Novel classes (used in few-shot fine-tuning): 
  - crazing
  - patches
  - pitted surface

The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1Somtykp_DwqGTe5by9PEfPDxPqYRFu-L).

## Installation

Follow the original installation instructions from the main [README.md](README.md) to install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

## Data Preparation

1. Download the IFDD dataset and place it in `./datasets/ifdd/`.

2. Process the dataset into COCO format:

```
# Process NEU-DET-METATRAIN dataset
python datasets/ifdd/3_prepare_meta_training.py --neu_det_path /path/to/NEU-DET-METATRAIN --output_path ./datasets/ifdd

# Process k-shot datasets for few-shot fine-tuning
python datasets/ifdd/2_gen_support_pool.py --data_path ./datasets/ifdd --k_shot 5 --num_sets 100
python datasets/ifdd/2_gen_support_pool.py --data_path ./datasets/ifdd --k_shot 10 --num_sets 100
python datasets/ifdd/2_gen_support_pool.py --data_path ./datasets/ifdd --k_shot 30 --num_sets 100
```

## Training and Evaluation Process

The training process follows three stages similar to the original Meta Faster R-CNN:

### 1. Meta-training

Meta-training on base classes (inclusion, rolled-in scales, scratches):

```
sh scripts/meta_training_ifdd_resnet101_multi_stages.sh
```

This script performs the three-step meta-training:
- First, train the baseline model
- Then add the feature fusion network in both Meta-RPN and Meta-Classifier
- Finally, add the attentive feature alignment

### 2. Training Base Classes Detection Head

Train a separate Faster R-CNN detection head for base classes:

```
sh scripts/faster_rcnn_with_fpn_ifdd_base_classes_branch.sh
```

### 3. Few-shot Fine-tuning

Perform 5/10/30-shot fine-tuning on novel classes (crazing, patches, pitted surface):

```
sh scripts/few_shot_finetune_ifdd_resnet101.sh
```

## Evaluation

After training, the model will be evaluated on the novel classes with the specified number of support shots.

## Configurations

All configuration files are in the `configs/ifdd` directory:
- `Base-IFDD-C4.yaml`: Base configuration for IFDD dataset
- `meta_training_ifdd_resnet101_stage_*.yaml`: Configurations for the three stages of meta-training
- `faster_rcnn_with_fpn_ifdd_base_classes_branch.yaml`: Configuration for training base classes
- `*shot_finetune_ifdd_resnet101.yaml`: Configurations for few-shot fine-tuning with different numbers of shots

## Credits

This adaptation is based on the following works:
- [Meta Faster R-CNN](https://arxiv.org/abs/2104.07719) by Han et al.
- [Few-Shot Steel Surface Defect Detection](https://ieeexplore.ieee.org/abstract/document/9623595) by Wang et al.

## About Meta Faster R-CNN

Meta Faster R-CNN, presented in the AAAI 2022 Oral paper "Meta Faster R-CNN: Towards Accurate Few-Shot Object Detection with Attentive Feature Alignment," is a powerful framework for few-shot object detection (FSOD). 

<div align="center"><img src="assets/figure_1.png" width="600"></div>

### Key Features of Meta Faster R-CNN

- **Natural Extension of Faster R-CNN**: The model extends the standard Faster R-CNN architecture to address few-shot scenarios using prototype-based metric learning.
- **Multi-Stage Training Process**: Training involves three distinct stages: baseline model training, feature fusion network integration, and attentive feature alignment.
- **Knowledge Preservation**: The model maintains knowledge of base classes while learning novel classes through a separate detection head.
- **Attentive Feature Alignment**: This mechanism effectively aligns features between query and support samples, improving detection accuracy with limited examples.
- **No Fine-tuning Requirement**: The meta-learning approach allows strong few-shot detection performance without extensive fine-tuning.

### Adaptation for IFDD Dataset

For steel surface defect detection, we've adapted Meta Faster R-CNN to work with the IFDD dataset by:

1. **Dataset Integration**: Modified the data loading pipeline to handle the IFDD dataset structure, which contains steel surface defect images in COCO format.
2. **Custom Evaluator**: Implemented the `IFDDEvaluator` that extends the COCO evaluator to provide metrics specific to steel defect detection.
3. **Support Feature Generation**: Adapted the support pool generation for the specific requirements of steel defect detection with 3 base classes and 3 novel classes.
4. **Shot Configuration**: Optimized for 5, 10, and 30-shot scenarios, which are most relevant for industrial defect detection applications.
5. **Specialized Configuration Files**: Created IFDD-specific configuration files that set appropriate parameters for the steel defect detection task.

This adaptation preserves the core strengths of Meta Faster R-CNN while tailoring it to the unique challenges of few-shot steel surface defect detection.

If you use the IFDD dataset, please cite:
```
@article{wang2021few,
  title={Few-Shot Steel Surface Defect Detection},
  author={Wang, Haohan and Li, Zhuoling and Wang, Haoqian},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2021},
  publisher={IEEE}
}
``` 
