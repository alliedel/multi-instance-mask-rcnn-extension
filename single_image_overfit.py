"""
(COMPLETE) The goal of single_image_overfit.py is to overfit the Mask R-CNN head to a single image.  This
ensures we can take training steps toward the gradient with respect to the ground truth provided by detectron2.
"""

import cv2
import numpy as np
import os
## Later objectives:
# Use this on the Kitchen dataset
# Change the training loss to re-learn the instance generation.
import torch
import torch.distributed

import vis_utils
from detectron2.evaluation.evaluator import inference_context
from detectron2.utils.events import EventStorage
from script_utils import FigExporter, get_custom_maskrcnn_cfg, run_evaluation, get_datapoint_file, \
    convert_datapoint_to_image_format
from trainer_apd import Trainer_APD

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def equal_ids(id1, id2):
    return str(id1).rstrip('0') == str(id2).rstrip('0')


def run_inference(trainer, inputs):
    previously_training = trainer.model.training
    trainer.model.eval()
    with inference_context(trainer.model), torch.no_grad():
        # Get proposals
        images = trainer.model.preprocess_image(inputs)
        features = trainer.model.backbone(images.tensor)
        proposalss, proposal_lossess = trainer.model.proposal_generator(images, features, None)

        # Get instance boxes, masks, and proposal idxs
        outputs, extra_proposal_details = trainer.model(inputs, trace_proposals=True)

    if previously_training:
        trainer.model.train()

    return {'outputs': outputs,
            'proposalss': proposalss,
            'proposal_lossess': proposal_lossess,
            'extra_proposal_details': extra_proposal_details
            }


def prep_image(datapoint, cfg):
    image_filename = datapoint['file_name']
    input_image = datapoint['image']
    input_image = np.asarray(input_image.permute(1, 2, 0)[:, :, [2, 1, 0]])
    input_image_from_file = cv2.imread(image_filename)
    input_image = convert_datapoint_to_image_format(input_image, input_image_from_file.shape[:2], cfg)
    return input_image


def train_on_single_image(trainer: Trainer_APD, datapoint, max_iters=100):
    for t in range(max_iters):
        with EventStorage(0) as trainer.storage:
            trainer.run_step_with_given_data(datapoint)


def main(image_id='486536', custom_head=False, max_iters=1, exporter=None):
    exporter = exporter or FigExporter()
    cfg = get_custom_maskrcnn_cfg()
    trainer = Trainer_APD(cfg)
    datapoint = torch.load(get_datapoint_file(cfg, image_id))
    if type(datapoint) is list:
        datapoint = datapoint[0]
    input_image = prep_image(datapoint, cfg)
    trainer_type = 'custom' if custom_head else 'standard'
    trainer.model.roi_heads.active_mask_head = trainer_type
    cfg_tag = f"_{trainer_type}"
    n_existing_exporter_images = len(exporter.generated_figures)

    batch = [datapoint]

    # Pre-training inference
    outputs_d = run_inference(trainer, batch)
    run_evaluation([input_image], cfg, outputs_d,
                   image_ids=[str(d['image_id']) + cfg_tag + '_pretrain' for d in batch],
                   model=trainer.model, exporter=exporter)
    figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag + '_pretrain' + '_collated'
    vis_utils.collate_figures(exporter.generated_figures[n_existing_exporter_images:], figure_name,
                              exporter=exporter)
    n_existing_exporter_images = len(exporter.generated_figures)

    # Train
    train_on_single_image(trainer, batch, max_iters=max_iters)

    # Check inference
    outputs_d = run_inference(trainer, batch)
    posttrain_tag = f'_posttrain_{max_iters}'
    run_evaluation([input_image], cfg, outputs_d,
                   image_ids=[str(d['image_id']) + cfg_tag + posttrain_tag for d in batch],
                   model=trainer.model, exporter=exporter)
    figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag + \
                  posttrain_tag + '_collated'
    vis_utils.collate_figures(exporter.generated_figures[n_existing_exporter_images:], figure_name,
                              exporter=exporter)

    vis_utils.show_groundtruth(datapoint, cfg)
    exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_groundtruth')


if __name__ == '__main__':
    exporter_ = FigExporter()
    image_ids_ = ['486536', '306284', '9']
    for max_iters_ in [1, 100]:
        main(image_id=image_ids_[2], max_iters=max_iters_, exporter=exporter_)
