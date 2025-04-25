# Meta Faster R-CNN with IFDD Dataset

This is an adaptation of Meta Faster R-CNN to work with the IFDD (Few-shot NEU-DET, FS-ND) dataset for few-shot steel surface defect detection.

*** Note that the current modifications are only effective on my device, and the configuration may differ on other devices. You may need to reconfigure. Additionally, due to differences in device performance, the final results may not be the same. ***

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
