import os

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata


_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_debug_train": ("coco/train2017", "coco/annotations/instances_coco_2017_debug_train.json"),
    "coco_2017_debug_val": ("coco/val2017", "coco/annotations/instances_coco_2017_debug_val.json"),
}


def register_debug_coco(root='datasets'):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


register_debug_coco()
