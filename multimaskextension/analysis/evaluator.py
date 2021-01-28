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
        self._predictions_per_mask_type = {
            m: [] for m in mask_names
        }

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        assert "instances" in outputs[0], NotImplementedError('We only handle instance segmentation evaluation.')
        for input, output in zip(inputs, outputs):
            # TODO this is ugly
            instances = output["instances"].to(self._cpu_device)
            rle_mask_names = []
            predictions = {}
            for mask_name in self.mask_names:
                prediction = {"image_id": input["image_id"]}
                if output['instances'].has(mask_name):
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
                    rle_mask_name = f"{mask_name}_rle"
                    setattr(instances, rle_mask_name, rles)
                    instances.remove(mask_name)
                else:
                    raise ValueError(f"{mask_name} not in model instances output")
                prediction["instances"] = instances_to_json(instances, input["image_id"],
                                                            mask_field_names=('pred_masks_rle',))
                if "proposals" in output:
                    prediction["proposals"] = output["proposals"].to(self._cpu_device)
                predictions[mask_name] = prediction
                self._predictions_per_mask_type[mask_name].append(prediction)

    def evaluate(self):
        for mask_name, predictions in self._predictions_per_mask_type.items():
            self._predictions = predictions
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
                    torch.save(predictions, f)

            self._results = OrderedDict()
            if "proposals" in self._predictions[0]:
                self._eval_box_proposals()
            if "instances" in self._predictions[0]:
                self._eval_predictions(set(self._tasks), mask_name)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, mask_name=None):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            task_name = task if mask_name is not None else task + '_' + mask_name
            self._results[task_name] = res


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


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
