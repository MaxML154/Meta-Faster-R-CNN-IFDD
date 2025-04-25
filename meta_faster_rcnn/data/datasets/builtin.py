"""
Created on 2025/4/25

@author: MaxML154
"""
import os


from .register_ifdd import register_all_ifdd
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import MetadataCatalog

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ifdd(_root)
