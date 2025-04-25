"""
Created on 2025/4/25

@author: MaxML154
"""
import os

from .register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import MetadataCatalog, DatasetCatalog

# ==== Predefined datasets and splits for IFDD ==========

def _get_ifdd_metadata():
    """Get IFDD metadata including classes for base and novel categories."""
    meta = {}
    # Base classes: inclusion, rolled-in scales, scratches
    # Novel classes: crazing, patches, pitted surface
    meta["base_classes"] = ["inclusion", "rolled-in_scales", "scratches"]
    meta["novel_classes"] = ["crazing", "patches", "pitted_surface"]
    meta["thing_classes"] = meta["base_classes"] + meta["novel_classes"]
    meta["thing_dataset_id_to_contiguous_id"] = {
        i + 1: i for i in range(len(meta["thing_classes"]))
    }
    return meta

_PREDEFINED_SPLITS_IFDD = {
    # Base training and validation sets
    "ifdd_base_train": ("ifdd/NEU-DET-METATRAIN/train2017", "ifdd/NEU-DET-METATRAIN/annotations/train2017.json"),
    "ifdd_novel_val": ("ifdd/NEU-DET-METATRAIN/val2017", "ifdd/NEU-DET-METATRAIN/annotations/val2017.json"),
    
    # Fewshot datasets
    "ifdd_novel_5shot": ("ifdd/5shot1-100/5shot1/train2017", "ifdd/5shot1-100/5shot1/annotations/train2017.json"),
    "ifdd_novel_10shot": ("ifdd/10shot1-100/10shot1/train2017", "ifdd/10shot1-100/10shot1/annotations/train2017.json"),
    "ifdd_novel_30shot": ("ifdd/30shot1-100/30shot1/train2017", "ifdd/30shot1-100/30shot1/annotations/train2017.json"),
}

def register_all_ifdd(root):
    """Register all IFDD datasets."""
    # Register additional shots with different seeds
    for shot in [5, 10, 30]:
        for seed in range(2, 101):  # We already have 1 above
            name = f"ifdd_novel_{shot}shot_seed{seed}"
            _PREDEFINED_SPLITS_IFDD[name] = (
                f"ifdd/{shot}shot1-100/{shot}shot{seed}/train2017", 
                f"ifdd/{shot}shot1-100/{shot}shot{seed}/annotations/train2017.json"
            )

    # Register all datasets
    metadata = _get_ifdd_metadata()
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_IFDD.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
        
        # Set evaluator type to ifdd instead of coco
        MetadataCatalog.get(key).evaluator_type = "ifdd" 