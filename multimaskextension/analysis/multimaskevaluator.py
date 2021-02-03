"""
Example of an evaluator (from DensePose) -- should be implemented by the MultiMask class
"""

import contextlib
import copy
import io
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator, instances_to_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate


class MultiMaskCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, mask_names=('pred_masks',),
                 eval_agg_masks=True, eval_distr_masks=True):
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
        self.eval_distr_masks = eval_distr_masks
        self.eval_agg_masks = eval_agg_masks

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
                    setattr(instances, 'pred_masks_rle', rles)
                    instances.remove(mask_name)
                else:
                    raise ValueError(f"{mask_name} not in model instances output")
                prediction["instances"] = instances_to_json(instances, input["image_id"])
                if "proposals" in output:
                    prediction["proposals"] = output["proposals"].to(self._cpu_device)
                predictions[mask_name] = prediction
                self._predictions_per_mask_type[mask_name].append(prediction)

    @property
    def mask_names_agg(self):
        return [nm for nm in self.mask_names if nm != 'pred_masks']

    def evaluate(self):
        all_results = OrderedDict()

        for mask_name in self._predictions_per_mask_type.keys():
            if self._distributed:
                comm.synchronize()
                self._predictions_per_mask_type[mask_name] = comm.gather(self._predictions[mask_name], dst=0)
                self._predictions_per_mask_type[mask_name] = \
                    list(itertools.chain(*self._predictions_per_mask_type[mask_name]))
                if not comm.is_main_process():
                    return {}
            if len(self._predictions_per_mask_type[mask_name]) == 0:
                self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
                return {}

        if self.eval_agg_masks:
            if self.eval_distr_masks:
                # We need to make a copy for agg since evaluate changes them.
                predictions_per_mask_type_for_agg = {
                    mask_name: copy.deepcopy(self._predictions_per_mask_type[mask_name])
                    for mask_name in self.mask_names_agg
                }
            else:
                predictions_per_mask_type_for_agg = self._predictions_per_mask_type
        else:
            predictions_per_mask_type_for_agg = None

        if self.eval_distr_masks:
            # Evaluate each mask type individually
            for mask_name, predictions in self._predictions_per_mask_type.items():
                self._predictions = predictions
                self._logger.info(f"\n\n**** Mask name: {mask_name}")
                print(f"\n\n**** Mask name: {mask_name}")
                if self._output_dir:
                    PathManager.mkdirs(self._output_dir)
                    file_path = os.path.join(self._output_dir, "instances_predictions.pth")
                    with PathManager.open(file_path, "wb") as f:
                        torch.save(predictions, f)

                self._results = OrderedDict()
                if "proposals" in self._predictions[0]:
                    self._eval_box_proposals()
                if "instances" in self._predictions[0]:
                    self._eval_predictions(set(self._tasks))
                res = copy.deepcopy(self._results)
                res_keys = list(res.keys())
                for k in res_keys:
                    all_results[k + '-' + mask_name] = res.pop(k)

            if len(self.mask_names) <= 1:
                return all_results

        if self.eval_agg_masks:
            # Evaluate all masks together
            # pred_masks is a pointer to another mask (pred_masks1, pred_masks2).  This could
            # change and be a source for bugs down the road, but the pointers aren't preserved after distributed compute,
            # so we can't verify at the moment
            n_val_images = len(self._predictions_per_mask_type[self.mask_names_agg[0]])
            # assert all([n_val_images == len(self._predictions_per_mask_type[nm]) for nm in mask_names_agg])
            predictions = [{'image_id': predictions_per_mask_type_for_agg[self.mask_names_agg[0]][i]['image_id'],
                            'instances': []}
                           for i in range(n_val_images)]

            # TODO(allie): definitely a faster way than this double for loop.
            for nm in self.mask_names_agg:
                subpreds = predictions_per_mask_type_for_agg[nm]
                for i in range(n_val_images):
                    assert subpreds[i]['image_id'] == predictions[i]['image_id']
                    predictions[i]['instances'].extend(subpreds[i]['instances'])
            mask_name = "agg-{}".format("_".join(self.mask_names_agg))
            self._logger.info(f"\n\n**** Mask name: {mask_name}")
            print(f"\n\n**** Mask name: {mask_name}")

            self._predictions = predictions
            if self._output_dir:
                PathManager.mkdirs(self._output_dir)
                file_path = os.path.join(self._output_dir, "instances_predictions.pth")
                with PathManager.open(file_path, "wb") as f:
                    torch.save(predictions, f)

            self._results = OrderedDict()
            if "proposals" in self._predictions[0]:
                self._eval_box_proposals()
            if "instances" in self._predictions[0]:
                self._eval_predictions(set(self._tasks))
            res = copy.deepcopy(self._results)
            res_keys = list(res.keys())
            for k in res_keys:
                all_results[k + mask_name] = res.pop(k)

        # Copy so the caller can do whatever with results
        return all_results
