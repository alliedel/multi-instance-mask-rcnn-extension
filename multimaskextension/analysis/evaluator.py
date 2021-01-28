"""
Example of an evaluator (from DensePose) -- should be implemented by the MultiMask class
"""

import copy
import itertools
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager


class MultiMaskCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, mask_names=('pred_masks',)):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir)
        self.mask_names = mask_names

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            # TODO this is ugly
            if "instances" in output:
                for mask_name in self.mask_names:
                    if output['instances'].has(mask_name):
                        instances = output["instances"].to(self._cpu_device)
                        # use RLE to encode the masks, because they are too large and takes memory
                        # since this evaluator stores outputs of the entire dataset
                        # Our model may predict bool array, but cocoapi expects uint8
                        rles = [
                            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                            for mask in getattr(instances, mask_name)
                        ]
                        for rle in rles:
                            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                            # json writer which always produces strings cannot serialize a bytestream
                            # unless you decode it. Thankfully, utf-8 works out (which is also what
                            # the pycocotools/_mask.pyx does).
                            rle["counts"] = rle["counts"].decode("utf-8")
                        setattr(instances, f"{mask_name}_rle", rles)
                        instances.remove(mask_name)
                    else:
                        raise ValueError(f"{mask_name} not in model instances output")
                    prediction["instances"] = instances_to_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "proposals" in self._predictions[0]:
            self._eval_box_proposals()
        if "instances" in self._predictions[0]:
            self._eval_predictions(set(self._tasks))
            self._eval_predictions(set(self._tasks))
            self._eval_predictions(set(self._tasks))

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


def instances_to_json(instances, img_id, mask_field_names=('pred_masks_rle',)):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    rles_multimask = {}
    has_mask = False
    for mask_field_name in mask_field_names:
        if instances.has(mask_field_names[0]):
            has_mask = True
            rles_multimask[mask_field_name] = getattr(instances, mask_field_name)
        else:
            raise ValueError(f"{mask_field_name} not in prediction")
    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            for mask_field_name in mask_field_names:
                has_mask = instances.has(mask_field_name)
                if has_mask:
                    rles_multimask[mask_field_name] = getattr(instances, mask_field_name)
                    result[f"segmentation_{mask_field_name}"] = rles_multimask[mask_field_name][k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results
