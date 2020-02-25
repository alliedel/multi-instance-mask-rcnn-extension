## The goal of this script is to load a pre-existing Mask R-CNN model and run it on COCO.
## Then, we will export images with the predicted bounding boxes.
## Then, we will export images with the proposal bounding boxes.
## Then, we will analyze the combination of GT and prediction boxes to see how many predictions include
# co-occurrence with another object.

import gc
import cv2
import os
import numpy as np
from PIL import Image
## Later objectives:
# Use this on the Kitchen dataset
# Change the training loss to re-learn the instance generation.
import torch
import torch.distributed
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.evaluator import inference_context

from script_utils import FigExporter, get_maskrcnn_cfg, DETECTRON_REPO, run_vanilla_evaluation, \
    run_evaluation, get_datapoint_file, collate_figures

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def run_vanilla_inference(predictor, inputs, train=False):
    if train is False:
        with inference_context(predictor.model), torch.no_grad():
            outputs = predictor.model(inputs)
    else:
        outputs = predictor.model(inputs)
    return outputs


def run_inference(predictor, inputs):
    with inference_context(predictor.model), torch.no_grad():
        # Get proposals
        images = predictor.model.preprocess_image(inputs)
        features = predictor.model.backbone(images.tensor)
        proposalss, proposal_lossess = predictor.model.proposal_generator(images, features, None)

        # Get instance boxes, masks, and proposal idxs
        outputs, extra_proposal_details = predictor.model(inputs, trace_proposals=True)

        return {'outputs': outputs,
                'proposalss': proposalss,
                'proposal_lossess': proposal_lossess,
                'extra_proposal_details': extra_proposal_details
                }


def convert_datapoint_to_image_format(img, out_shape, cfg):
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if cfg.INPUT.FORMAT == "BGR":
        img = np.asarray(img[:, :, [2, 1, 0]])
    else:
        img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
    if cfg.INPUT.FORMAT == "BGR":
        img = np.asarray(img[:, :, [2, 1, 0]])
    else:
        img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
    img = cv2.resize(img, out_shape[::-1])
    return img


def main(config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
         image_id='486536', flip_lr=False, exporter=None):
    if exporter is None:
        exporter = exporter_
    n_existing_exporter_images = len(exporter.generated_figures)
    cfg = get_maskrcnn_cfg(config_filepath)
    predictor = DefaultPredictor(cfg)
    # predictor.model.training = True

    saved_input_file = get_datapoint_file(cfg, image_id)
    datapoint = torch.load(saved_input_file)
    if type(datapoint) is not list:
        datapoint = [datapoint]
    if flip_lr:
        assert all(d['image'].shape[0] == 3 for d in datapoint)
        for d in datapoint:
            d['image'] = d['image'].flip(dims=(2,))
    image_filenames = [d['file_name'] for d in datapoint]
    input_images = [d['image'] for d in datapoint]
    input_images = [np.asarray(img.permute(1, 2, 0)[:, :, [2, 1, 0]]) for img in input_images]
    input_images_from_files = [cv2.imread(fn) for fn in image_filenames]
    input_images = [convert_datapoint_to_image_format(im, im2.shape[:2], cfg)
                    for im, im2 in zip(input_images, input_images_from_files)]

    output = run_vanilla_inference(predictor, datapoint, train=False)
    run_vanilla_evaluation(input_images, cfg, output, image_ids=[str(d['image_id']) + ('_flip' if flip_lr else '')
                                                                 for d in datapoint], model=predictor.model)

    n_existing_exporter_images = len(exporter.generated_figures)
    outputs_d = run_inference(predictor, datapoint)
    run_evaluation(input_images, cfg, outputs_d,
                   image_ids=[str(d['image_id']) + ('_flip' if flip_lr else '') for d in datapoint],
                   model=predictor.model)
    my_image_ids = [str(d['image_id']) + ('_flip' if flip_lr else '') for d in datapoint]
    for my_image_id in my_image_ids:
        figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + my_image_id + '_collated'
        collate_figures(exporter.generated_figures[n_existing_exporter_images:], figure_name, exporter=exporter)


if __name__ == '__main__':
    exporter_ = FigExporter()
    image_ids = ['486536', '306284', '9']
    for image_id in image_ids:
        for flip_lr in [False, True]:
            main(image_id=image_id, flip_lr=flip_lr, exporter=exporter_)
            gc.collect()
