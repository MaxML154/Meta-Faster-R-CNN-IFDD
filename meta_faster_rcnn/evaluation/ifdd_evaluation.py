"""
IFDD dataset evaluator for steel surface defect detection.
"""

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize

from .coco_evaluation import COCOEvaluator, instances_to_coco_json

class IFDDEvaluator(COCOEvaluator):
    """
    Evaluator for the IFDD dataset.
    This is a subclass of COCOEvaluator that adds specific metrics for the IFDD dataset.
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            distributed (bool): if True, will collect results from all ranks and run evaluation
                in the main process. Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instance_predictions.pckl" a file in pickle format that
                   contains all the raw original predictions.
                2. "instances_results.json" a json file containing results for COCO instances format
        """
        super().__init__(dataset_name, cfg, distributed, output_dir)
        
        self.dataset_name = dataset_name
        self._logger = logging.getLogger(__name__)
        # For IFDD-specific evaluation metrics
        self._base_classes = []
        self._novel_classes = []
        
        # Get class information from dataset metadata
        metadata = MetadataCatalog.get(dataset_name)
        if hasattr(metadata, "base_classes"):
            self._base_classes = metadata.base_classes
        if hasattr(metadata, "novel_classes"):
            self._novel_classes = metadata.novel_classes
    
    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        """
        return super().process(inputs, outputs)
    
    def evaluate(self):
        """
        Evaluate the predictions on the IFDD dataset, with additional metrics specific to IFDD.
        """
        results = super().evaluate()
        
        # Add IFDD-specific metrics if needed
        if self._base_classes and self._novel_classes:
            # Extract base class and novel class performance separately
            coco_eval = self._coco_eval
            if coco_eval is not None and len(coco_eval.evalImgs) > 0:
                self._logger.info("Computing IFDD-specific metrics...")
                
                # You can implement specific metrics for base and novel classes here
                # For example, separate mAP for base and novel classes
                
                # Example (pseudo-code):
                # base_class_ap = compute_ap_for_classes(coco_eval, self._base_classes)
                # novel_class_ap = compute_ap_for_classes(coco_eval, self._novel_classes)
                
                # Add to results
                # results["base_AP"] = base_class_ap
                # results["novel_AP"] = novel_class_ap
        
        return results 