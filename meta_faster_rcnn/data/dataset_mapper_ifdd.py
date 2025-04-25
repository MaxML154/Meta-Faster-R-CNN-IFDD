"""
Created on 2025/4/25

@author: MaxML154
"""
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import pandas as pd
from detectron2.data.catalog import MetadataCatalog

"""
This file contains the dataset mapper for the IFDD dataset.
"""

__all__ = ["DatasetMapperWithSupportIFDD"]


class DatasetMapperWithSupportIFDD:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.few_shot       = cfg.INPUT.FS.FEW_SHOT
        self.support_way    = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot   = cfg.INPUT.FS.SUPPORT_SHOT
        self.seeds          = cfg.DATASETS.SEEDS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        if self.is_train:
            # support_df
            self.support_on = True
            if self.few_shot:
                # Load appropriate support data based on shot and seed
                if self.seeds == 0:
                    shot_path = f"./datasets/ifdd/{self.support_shot}shot_support_df.pkl"
                    self.support_df = pd.read_pickle(shot_path)
                    print(f"training support_df={shot_path}")
                else:
                    seed_path = f"./datasets/ifdd/seed{self.seeds}/{self.support_shot}shot_support_df.pkl"
                    self.support_df = pd.read_pickle(seed_path)
                    print(f"training support_df={seed_path}")
            else:
                # For meta-training
                self.support_df = pd.read_pickle("./datasets/ifdd/metatrain_support_df.pkl")
                print("training support_df=./datasets/ifdd/metatrain_support_df.pkl")

            # Get IFDD dataset metadata
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            
            # If we need to map category IDs
            if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
                reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]
                self.support_df['category_id'] = self.support_df['category_id'].map(reverse_id_mapper)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        if self.is_train:
            # support
            if self.support_on:
                if "annotations" in dataset_dict:
                    # USER: Modify this if you want to keep them for some reason.
                    for anno in dataset_dict["annotations"]:
                        if not self.mask_on:
                            anno.pop("segmentation", None)
                        if not self.keypoint_on:
                            anno.pop("keypoints", None)
                support_images, support_bboxes, support_cls = self.generate_support(dataset_dict)
                dataset_dict['support_images'] = torch.as_tensor(np.ascontiguousarray(support_images))
                dataset_dict['support_bboxes'] = support_bboxes
                dataset_dict['support_cls'] = support_cls

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.proposal_min_box_size,
                self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)
            
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

    def generate_support(self, dataset_dict):
        support_way = self.support_way  # For IFDD this would be the number of classes
        support_shot = self.support_shot  # Number of examples per class
        
        # For IFDD, we get either the image ID or direct ID from the dataset_dict
        query_id = dataset_dict['annotations'][0]['id'] if 'id' in dataset_dict['annotations'][0] else dataset_dict['annotations'][0]['category_id']
        query_img = dataset_dict['image_id']
        
        # Get category ID of the query image
        if 'category_id' in dataset_dict['annotations'][0]:
            query_cls = dataset_dict['annotations'][0]['category_id']
        else:
            # If no category_id in annotations, try to get it from support_df
            query_cls = self.support_df.loc[self.support_df['id'] == query_id, 'category_id'].tolist()[0]
        
        # Get all class IDs in this image (for multi-object images)
        all_cls = self.support_df.loc[self.support_df['image_id'] == query_img, 'category_id'].tolist()
        
        # Initialize arrays for support data
        support_data_all = np.zeros((support_way * support_shot, 3, 320, 320), dtype=np.float32)
        support_box_all = np.zeros((support_way * support_shot, 4), dtype=np.float32)
        
        # Keep track of used images and IDs to avoid duplicates
        used_image_id = [query_img]
        used_id_ls = []
        
        # Get unique category IDs in the current image
        used_category_id = list(set(all_cls))
        support_category_id = []
        mixup_i = 0

        # For each shot (example) we need
        for shot in range(support_shot):
            # Find support examples with the same class but different images
            support_list = self.support_df.loc[
                (self.support_df['category_id'] == query_cls) & 
                (~self.support_df['image_id'].isin(used_image_id)) & 
                (~self.support_df['id'].isin(used_id_ls)), 
                'id'
            ]
            
            # Skip if no matching support examples found
            if support_list.empty:
                continue
                
            # Randomly sample one support example
            support_id = support_list.sample().tolist()[0]
            
            # Get class and image ID for this support example
            support_cls = self.support_df.loc[self.support_df['id'] == support_id, 'category_id'].tolist()[0]
            support_img = self.support_df.loc[self.support_df['id'] == support_id, 'image_id'].tolist()[0]
            
            # Mark this example as used
            used_id_ls.append(support_id)
            
            # Get the row from the support dataframe
            support_db = self.support_df.loc[self.support_df['id'] == support_id, :]
            assert support_db['id'].values[0] == support_id
            
            # Get support image and box
            support_data = utils.read_image(support_db['file_name'].values[0], format=self.img_format)
            
            # Extract the support box
            support_box = support_db['bbox'].values[0]
            support_box = np.array(support_box)
            
            # Ensure the box coordinates are valid
            x1, y1, x2, y2 = support_box
            img_h, img_w = support_data.shape[0], support_data.shape[1]
            x1 = np.clip(x1, 0, img_w - 1)
            y1 = np.clip(y1, 0, img_h - 1)
            x2 = np.clip(x2, x1 + 1, img_w)
            y2 = np.clip(y2, y1 + 1, img_h)
            support_box = np.array([x1, y1, x2, y2])
            
            # Crop and resize support image to a fixed size (320x320)
            support_data = support_data[int(y1):int(y2), int(x1):int(x2), :]
            support_data = np.asarray(Image.fromarray(support_data).resize((320, 320)))
            
            # Normalize support image (similar to the query image)
            support_data = support_data.astype(np.float32, copy=False)
            support_data = support_data / 255.
            support_data = support_data.transpose(2, 0, 1)
            
            # Adjust the support box for the resized image
            support_box_new = np.array([0, 0, 320, 320])
            
            # Store this support example
            support_data_all[mixup_i] = support_data
            support_box_all[mixup_i] = support_box_new
            support_category_id.append(support_cls)
            mixup_i += 1

        # If we couldn't find enough distinct support images, duplicate the existing ones
        while mixup_i < support_way * support_shot:
            support_data_all[mixup_i] = support_data_all[mixup_i % (mixup_i + 1)]
            support_box_all[mixup_i] = support_box_all[mixup_i % (mixup_i + 1)]
            support_category_id.append(support_category_id[mixup_i % (mixup_i + 1)])
            mixup_i += 1

        return support_data_all, support_box_all, support_category_id 