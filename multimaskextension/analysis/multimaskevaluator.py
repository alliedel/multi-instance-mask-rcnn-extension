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
import time
from collections import OrderedDict
from pycocotools import mask as maskUtils

import detectron2.utils.comm as comm
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator, COCOeval
# from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco, instances_to_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate


def xor(a, b):
    return (a or b) and (not (a and b))


def log_and_print(logger, *args, **kwargs):
    if logger is not None:
        logger.info(*args, **kwargs)
    print(*args, **kwargs)


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
        # self.reset()
        self.mask_names = mask_names
        self._predictions_per_mask_type = {
            m: [] for m in mask_names
        }
        assert eval_distr_masks is True, NotImplementedError
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
                prediction["instances"] = instances_to_json(instances, input["image_id"],
                                                            in_bbox_mode=BoxMode.XYXY_ABS,
                                                            out_bbox_mode=BoxMode.XYXY_ABS)
                if "proposals" in output:
                    prediction["proposals"] = output["proposals"].to(self._cpu_device)
                predictions[mask_name] = prediction
                self._predictions_per_mask_type[mask_name].append(prediction)

    @property
    def mask_names_agg(self):
        return [nm for nm in self.mask_names if nm != 'pred_masks']

    def evaluate(self):
        # I. Prepare predictions
        #   A. Wait for distributed inference to finish; Collect predictions
        for mask_name in self._predictions_per_mask_type.keys():
            if self._distributed:
                comm.synchronize()
                self._predictions_per_mask_type[mask_name] = comm.gather(self._predictions[mask_name], dst=0)
                self._predictions_per_mask_type[mask_name] = \
                    list(itertools.chain(*self._predictions_per_mask_type[mask_name]))
                if not comm.is_main_process():
                    return OrderedDict()
            if len(self._predictions_per_mask_type[mask_name]) == 0:
                self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
                return OrderedDict()

        #   B. Remove masks with area 0
        log_and_print(self._logger, f"\n\n**** Removing zero-area instances ")
        for mask_name in self.mask_names_agg:
            empty_mask_idxs_per_img = self.find_empty_masks(self._predictions_per_mask_type[mask_name])
            instances_removed = self.verify_and_remove_instances(empty_mask_idxs_per_img,
                                                                 self._predictions_per_mask_type[mask_name])
            log_and_print(self._logger, f"\n Mask {mask_name}: Removed {len(instances_removed)} zero-area instances.")

        predictions_per_mask_type_for_agg = {
            mask_name: copy.deepcopy(self._predictions_per_mask_type[mask_name])
            for mask_name in self.mask_names_agg
        } if self.eval_agg_masks else None
        #   C. Make a copy for aggregated evaluation stage (evaluate() modifies predictions)

        # II. Per-mask evaluation & fill predictions['instances'][inst_idx]['area'] and ['id']
        log_and_print(self._logger, "Evaluating multimask predictions")
        predictions_per_mask_type = self._predictions_per_mask_type
        all_individual_results, coco_evals_individual = self.eval_predictions_multimask(
            predictions_per_mask_type, self._coco_api)

        if len(self.mask_names) <= 1:
            return all_individual_results, coco_evals_individual

        if self.eval_agg_masks:
            log_and_print(self._logger, "\n\n *** Running aggregated mask evaluation")
            agg_results, coco_evals_agg = self.aggregate_and_evaluate(predictions_per_mask_type_for_agg,
                                                                      self._coco_api)
        else:
            agg_results = OrderedDict()
            coco_evals_agg = OrderedDict()
        all_results = all_individual_results
        all_results.update(agg_results)
        all_coco_evals = coco_evals_individual
        all_coco_evals.update(coco_evals_agg)
        torch.save(all_results, os.path.join(self._output_dir, "statresults.pth"))
        torch.save(all_coco_evals, os.path.join(self._output_dir, "cocoevals.pth"))
        return copy.deepcopy(all_results), copy.deepcopy(all_coco_evals)

    def verify_and_remove_instances(self, mask_idxs_to_remove_per_img, predictions_to_remove_from, ref_preds=None):
        instances_removed = []
        for imgidx, empty_idxs in enumerate(mask_idxs_to_remove_per_img):
            for empty_idx in sorted(empty_idxs)[::-1]:
                removed_inst = predictions_to_remove_from[imgidx]['instances'].pop(empty_idx)
                if ref_preds is not None:
                    assert self.is_same_inst(ref_preds[imgidx]['instances'][empty_idx], removed_inst)
                instances_removed.append({'imgidx': imgidx,
                                          'image_id': predictions_to_remove_from[imgidx]['image_id'],
                                          'inst_idx': empty_idx,
                                          'inst_id': ref_preds[imgidx]['instances'][empty_idx]['id']
                                          if ref_preds is not None else None,
                                          'area': ref_preds[imgidx]['instances'][empty_idx]['area']
                                          if ref_preds is not None else None}
                                         )
        return instances_removed

    def instlstidx_to_perimgidx(self, perimg_instances):
        """
        Predictions are nested by image; coco_results are not.  Converting indices between is useful.
        """
        absidx_to_ij = {}
        absidx = 0
        for i, p in enumerate(perimg_instances):
            for j in range(len(p['instances'])):
                absidx_to_ij[absidx] = (i, j)
                absidx += 1
        return absidx_to_ij

    def aggregate_and_evaluate(self, predictions_per_mask_type_for_agg, coco_api):
        # Step 1: Aggregate all masks; compute per-image, per-instance matching/scoring
        # pred_masks is a pointer to another mask (pred_masks1, pred_masks2).  This could
        # change and be a source for bugs down the road, but the pointers aren't preserved
        # after distributed compute, so we can't verify at the moment
        n_val_images = len(self._predictions_per_mask_type[self.mask_names_agg[0]])
        # TODO(allie): definitely a faster way than this double for loop.
        predictions = [{'image_id': predictions_per_mask_type_for_agg[self.mask_names_agg[0]][i]['image_id'],
                        'instances': []}
                       for i in range(n_val_images)]
        for nm in self.mask_names_agg:
            subpreds = predictions_per_mask_type_for_agg[nm]
            for i in range(n_val_images):
                assert subpreds[i]['image_id'] == predictions[i]['image_id']
                predictions[i]['instances'].extend(subpreds[i]['instances'])

        predictions_saved = copy.deepcopy(predictions)
        coco_results = self.predictions_to_coco_results(predictions)
        predictions = predictions_saved

        n_unfiltered = len(coco_results)
        coco_results_for_filt = copy.deepcopy(coco_results)
        dts_to_remove = get_disadvantaged_dt_ids_to_remove(coco_api, coco_results_for_filt, 'segm',
                                                           kpt_oks_sigmas=self._kpt_oks_sigmas)
        abs_to_perimg = self.instlstidx_to_perimgidx(predictions)
        pred_idxs_to_remove = [abs_to_perimg[x] for x in dts_to_remove]
        pidxs = [[] for _ in range(len(predictions))]
        for (pi, pj) in pred_idxs_to_remove:
            pidxs[pi].append(pj)
        for p, pidxs in zip(predictions, pidxs):
            for pi in sorted(pidxs, reverse=True):
                p['instances'].pop(pi)

        coco_results[:] = [c for i, c in enumerate(coco_results) if i not in set(dts_to_remove)]
        assert len(coco_results) == n_unfiltered - len(dts_to_remove), 'Debug error'
        assert sum(len(p['instances']) for p in predictions) == len(coco_results), 'Debug error'

        mask_name = "agg-{}".format("_".join(self.mask_names_agg))

        log_and_print(self._logger, f"\n\n**** Mask name: {mask_name}")
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, f"instances_predictions_{mask_name}.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)
        self._results = None
        res_unnamed, coco_evals_unnamed = self.eval_single_set_of_predictions(
            predictions, coco_api, self._tasks, output_dir=self._output_dir,
            json_outfile=f"coco_instances_results_{mask_name}.json")
        combined_results = OrderedDict()
        coco_evals = OrderedDict()
        res_keys = list(res_unnamed.keys())
        for k in res_keys:
            combined_results[k + '-' + mask_name] = res_unnamed.pop(k)
        for task in list(coco_evals_unnamed.keys()):
            coco_evals[(mask_name, task)] = coco_evals_unnamed.pop(task)
        return combined_results, coco_evals

    def eval_single_set_of_predictions(self, predictions, coco_api, tasks, output_dir,
                                       json_outfile="coco_instances_results.json"):
        """
        Note: self._results will be populated in the middle of this.
        """
        coco_evals = {}
        try:
            assert self._results is None or len(self._results) == 0
        except AttributeError:
            pass  # Fine if self._results is not populated
        self._results = OrderedDict()
        self._predictions = predictions

        if "proposals" in predictions[0]:
            self._eval_box_proposals()
        if "instances" in predictions[0]:
            tasks = set(tasks)
            log_and_print(self._logger, "Preparing results for COCO format ...")
            coco_results = self.predictions_to_coco_results(predictions)

            if output_dir:
                file_path = os.path.join(output_dir, json_outfile)
                log_and_print(self._logger, "Saving results to {}".format(file_path))
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(coco_results))
                    f.flush()

            if not self._do_evaluation:
                log_and_print(self._logger, "Annotations are not available for evaluation.")
                return
            log_and_print(self._logger, "Evaluating predictions ...")

            for task in sorted(tasks):
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas, summarize=True
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results(
                    coco_eval, task, class_names=self._metadata.get("thing_classes")
                )
                self._results[task] = res
                coco_evals[task] = coco_eval

        res = copy.deepcopy(self._results)
        self._results = None  # Making sure we don't access this way (modifying within this function is silly to me)
        return res, coco_evals

    def predictions_to_coco_results(self, predictions):
        # Recommend running this before call, as it changes 'predictions'
        # predictions = copy.deepcopy(predictions)
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        return coco_results

    def eval_predictions_multimask(self, multimask_predictions_dict, coco_api):
        all_results = OrderedDict()
        # Evaluate each mask type individually
        coco_evals = {}
        for mask_name, predictions in multimask_predictions_dict.items():
            log_and_print(self._logger, f"\n\n**** Mask name: {mask_name}")
            print(f"\n\n**** Mask name: {mask_name}")
            if self._output_dir:
                PathManager.mkdirs(self._output_dir)
                file_path = os.path.join(self._output_dir, f"instances_predictions_{mask_name}.pth")
                with PathManager.open(file_path, "wb") as f:
                    torch.save(predictions, f)
            self._results = None
            res, coco_evals_mask = self.eval_single_set_of_predictions(
                predictions, coco_api, self._tasks, output_dir=self._output_dir,
                json_outfile=f"coco_instances_results_{mask_name}.json")
            res_keys = list(res.keys())
            for k in res_keys:
                all_results[k + '-' + mask_name] = res.pop(k)
            for task in list(coco_evals_mask.keys()):
                coco_evals[(mask_name, task)] = coco_evals_mask.pop(task)
        return all_results, coco_evals

    @staticmethod
    def find_empty_masks(evaluated_predictions_per_img):
        """
        We use the fast method (compare encoded masks to an encoded empty mask).

        The slow (and more natural) method that would enable actual area compute (so we could threshold below 10,
        for instance) is slower, I believe.  The comparison through my IDE, however, did not sow that.:
        FAST method:
        %timeit [np.array([i for i, inst in enumerate(imeval['instances']) if inst['segmentation']['counts'] ==
        zero_mask_rle]) for imeval in evaluated_predictions_per_img]
        2 ms ± 4.38 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        SLOW method:
        %timeit [np.arange(0, len(imeval['instances']), dtype=int)[np.array([mask_util.decode(inst[
        'segmentation']).sum() for i, inst in enumerate(imeval['instances'])]) == 0] for imeval in
        evaluated_predictions_per_img]
        17.2 s ± 2.28 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """
        zero_mask = mask_util.decode(evaluated_predictions_per_img[0]['instances'][0]['segmentation'])
        zero_mask[:] = 0
        zero_mask_rle = mask_util.encode(zero_mask)['counts'].decode()  # last decode() converts to Unicode - needed for
        empty_mask_idxs_per_img = [
            np.array([i for i, inst in enumerate(imeval['instances'])
                      if inst['segmentation']['counts'] == zero_mask_rle])
            for imeval in evaluated_predictions_per_img]
        return empty_mask_idxs_per_img

    @staticmethod
    def is_same_inst(inst1, inst2):
        if hasattr(inst1, 'id') and hasattr(inst2, 'id'):
            if inst1['id'] != inst2['id']:
                return False
        if inst1['score'] != inst2['score']:
            return False
        if inst1['image_id'] != inst2['image_id']:
            return False
        if not all(x == y for x, y in zip(inst1['bbox'], inst2['bbox'])):
            return False
        return True


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None, summarize=True):
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
    coco_eval = cocoeval_evaluate(coco_eval)
    if summarize:
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval


def get_disadvantaged_dt_ids_to_remove(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_results_orig = copy.deepcopy(coco_results)
    if iou_type == "segm":
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
    p = coco_eval.params
    # add backward compatibility if useSegm is specified in params
    if not p.useSegm is None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    coco_eval.params = p

    coco_eval._prepare()

    tic = time.time()
    print('Running IOU computation...')
    catIds = p.catIds if p.useCats else [-1]
    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = coco_eval.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = coco_eval.computeOks

    print('Finding detections to filter out...')
    dtids_to_remove = {}
    for imgId in p.imgIds:
        for catId in catIds:
            if p.useCats:
                gt = coco_eval._gts[imgId, catId]
                dt = coco_eval._dts[imgId, catId]
            else:
                gt = [_ for cId in p.catIds for _ in coco_eval._gts[imgId, cId]]
                dt = [_ for cId in p.catIds for _ in coco_eval._dts[imgId, cId]]
            ious_imcat = computeIoU(imgId, catId)  # if len(ious_imcat) == 0: continue
            dtids_to_remove[(imgId, catId)] = identify_dts_to_filter_by_iou(ious_imcat, gt, dt, coco_eval.params)

    coco_idxs_to_remove = [i for i, c in enumerate(coco_results) if (c['id'] in dtids_to_remove[(c['image_id'],
                                                                                                 c['category_id'])])]
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))
    assert all(c1['score'] == c2['score'] for c1, c2 in zip(coco_results_orig, coco_results)), 'Debug error'
    return coco_idxs_to_remove


def identify_dts_to_filter_by_iou(ious_imcat, gt_imcat, dt_imcat, cocoeval_params):
    # Use all areas
    aRng = cocoeval_params.areaRng[0]
    assert all(aRng[0] <= a[0] and aRng[1] >= a[1] for a in cocoeval_params.areaRng)
    maxDet = cocoeval_params.maxDets[-1]

    if np.sum(ious_imcat) == 0:
        return np.array([])
    p = cocoeval_params
    if len(gt_imcat) == 0 and len(dt_imcat) == 0:
        return None

    for g in gt_imcat:
        if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt_imcat], kind='mergesort')
    gt = [gt_imcat[i] for i in gtind]
    dtind = np.argsort([-d['score'] for d in dt_imcat], kind='mergesort')
    dt = [dt_imcat[i] for i in dtind[0:maxDet]]
    # load computed ious
    ious = ious_imcat[:, gtind] if len(ious_imcat) > 0 else ious_imcat
    T = len(p.iouThrs)
    dtIg, dtm, gtIg, gtm, gtmiou = match_dt_gt(dt, gt, ious)
    if (dtm > 0).all():  # all dt assigned
        return []

    # Report unmatched dt that are 'forgiven' because they would have been matched to an instance outside of their bbox
    dt_to_remove = []
    for dti in (dtm == 0).nonzero()[0]:
        if ious[dti, :].sum() == 0:  # no overlap with any GT; must keep as FP
            continue
        closest_gt_match_idx = np.argmax(ious[dti, :])
        competing_dt = [d for d in dt if d['id'] == gtm[closest_gt_match_idx]][0]
        # Make sure bbox isn't the same (not forgiven for competing against masks inside your bbox)
        if not bboxes_match(competing_dt, dt[dti]):
            dt_to_remove.append(dt[dti]['id'])

    return dt_to_remove


def bboxes_match(inst1, inst2):
    return all([x == y for x, y in zip(inst1['bbox'], inst2['bbox'])])


def match_dt_gt(dt, gt, ious):
    iscrowd = [int(o['iscrowd']) for o in gt]
    G = len(gt)
    D = len(dt)
    gtm = np.zeros(G)
    gtmiou = np.zeros(G)
    dtm = np.zeros(D)
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros(D)
    if not len(ious) == 0:
        for dind, d in enumerate(dt):
            # information about best match so far (m=-1 -> unmatched)
            m = -1
            iou = 1e-10
            for gind, g in enumerate(gt):
                # if this gt already matched, and not a crowd, continue
                if gtm[gind] > 0 and not iscrowd[gind]:
                    continue
                # if dt matched to reg gt, and on ignore gt, stop
                if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                    break
                # continue to next gt unless better match made
                if ious[dind, gind] < iou:
                    continue
                # if match successful and best so far, store appropriately
                iou = ious[dind, gind]
                m = gind
            # if match made store id of match for both dt and gt
            if m == -1:
                continue
            dtIg[dind] = gtIg[m]
            dtm[dind] = gt[m]['id']
            gtm[m] = d['id']
            gtmiou[m] = iou
    return dtIg, dtm, gtIg, gtm, gtmiou


def cocoeval_evaluate(cocoeval: COCOeval):
    '''
    NOTE(allie): REPRODUCING FUNCTION OUTSIDE OF CLASS
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # evaluateImg = cocoeval.evaluateImg
    tic = time.time()
    print('Running per image evaluation...')
    p = cocoeval.params
    # add backward compatibility if useSegm is specified in params
    if not p.useSegm is None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    cocoeval.params = p

    cocoeval._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = cocoeval.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = cocoeval.computeOks
    cocoeval.ious = {(imgId, catId): computeIoU(imgId, catId) \
                     for imgId in p.imgIds
                     for catId in catIds}

    maxDet = p.maxDets[-1]
    cocoeval.evalImgs = [evaluateImg(cocoeval, imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds
                         ]
    cocoeval._paramsEval = copy.deepcopy(cocoeval.params)
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))
    return cocoeval


def evaluateImg(cocoeval: COCOeval, imgId, catId, aRng, maxDet):
    '''
    NOTE(allie): REPRODUCING FUNCTION OUTSIDE OF CLASS
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    p = cocoeval.params
    if p.useCats:
        gt = cocoeval._gts[imgId, catId]
        dt = cocoeval._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in cocoeval._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in cocoeval._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return None

    for g in gt:
        if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = cocoeval.ious[imgId, catId][:, gtind] if len(cocoeval.ious[imgId, catId]) > 0 else cocoeval.ious[
        imgId, catId]

    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    gtmiou = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T, D))
    if not len(ious) == 0:
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d['id']
                gtmiou[tind, m] = iou
    # set unmatched detections outside of area range to ignore
    a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
    # store results for given image and category
    return {
        'image_id': imgId,
        'category_id': catId,
        'aRng': aRng,
        'maxDet': maxDet,
        'dtIds': [d['id'] for d in dt],
        'gtIds': [g['id'] for g in gt],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'gtMatchIous': gtmiou,
        'dtScores': [d['score'] for d in dt],
        'gtIgnore': gtIg,
        'dtIgnore': dtIg,
    }


def instances_to_json(instances, img_id, in_bbox_mode=BoxMode.XYXY_ABS, out_bbox_mode=BoxMode.XYXY_ABS):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    if in_bbox_mode != out_bbox_mode:
        boxes = BoxMode.convert(boxes, in_bbox_mode, out_bbox_mode)
        # boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks_rle")
    if has_mask:
        rles = instances.pred_masks_rle

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
            # "box_mode": out_bbox_mode
        }
        if has_mask:
            result["segmentation"] = rles[k]
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
