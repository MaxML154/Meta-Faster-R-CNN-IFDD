"""
Modified on 2025/4/25

@author: MaxML154
"""
from .coco_evaluation import COCOEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .ifdd_evaluation import IFDDEvaluator
import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import MetadataCatalog

__all__ = [k for k in globals().keys() if not k.startswith("_")]

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = cfg.OUTPUT_DIR
    
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    
    if evaluator_type == "ifdd":
        return IFDDEvaluator(dataset_name, cfg, comm.is_main_process(), output_folder)
    elif evaluator_type == "coco":
        return COCOEvaluator(dataset_name, cfg, comm.is_main_process(), output_folder)
    
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "No evaluator implementation for the dataset {}".format(dataset_name)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    
    return DatasetEvaluators(evaluator_list)
