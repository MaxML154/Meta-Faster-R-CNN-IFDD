from . import builtin  # ensure the builtin datasets are registered
from .register_coco import register_coco_instances
from .register_ifdd import register_all_ifdd

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
