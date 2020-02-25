## The goal of this script is to load a pre-existing Mask R-CNN model and run it on COCO.
## Then, we will export images with the predicted bounding boxes.
## Then, we will export images with the proposal bounding boxes.
## Then, we will analyze the combination of GT and prediction boxes to see how many predictions include
# co-occurrence with another object.

import cv2
import numpy as np
import os
## Later objectives:
# Use this on the Kitchen dataset
# Change the training loss to re-learn the instance generation.
import torch
import torch.distributed

from detectron2.modeling.roi_heads.multi_instance_roi_heads import MultiROIHeadsAPD
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation.evaluator import inference_context
from script_utils import FigExporter, get_maskrcnn_cfg, get_custom_maskrcnn_cfg, DETECTRON_REPO, \
    run_evaluation, get_datapoint_file, collate_figures, convert_datapoint_to_image_format

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def equal_ids(id1, id2):
    return str(id1).rstrip('0') == str(id2).rstrip('0')


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


def build_dataloader(cfg):
    # dataloaders_eval = {
    #     'val': DefaultTrainer.build_test_loader(cfg, {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s])
    #     for s
    #     in ('train', 'val')
    # }
    train_dataloader = DefaultTrainer.build_train_loader(cfg)
    return train_dataloader


def main(config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
         image_ids=('486536',), flip_lr=False, exporter=exporter_):
    if type(image_ids) is str:
        image_ids = [image_ids]
    cfg = get_maskrcnn_cfg()
    custom_cfg = get_custom_maskrcnn_cfg()
    predictors = {
        'custom': DefaultPredictor(custom_cfg),
        'standard': DefaultPredictor(cfg)
    }
    assert isinstance(predictors['custom'].model.roi_heads, MultiROIHeadsAPD)
    predictors['custom'].model.roi_heads.active_mask_head_name = 'standard'
    if isinstance(predictors['standard'].model.roi_heads, MultiROIHeadsAPD):
        predictors['standard'].model.roi_heads.active_mask_head_name = 'standard'

    rls = [False, True] if flip_lr is True else [False]
    for rl in rls:
        for image_id in image_ids:
            saved_input_file = get_datapoint_file(cfg, image_id)
            datapoint = torch.load(saved_input_file)
            if type(datapoint) is not list:
                datapoint = [datapoint]
            if rl:
                assert all(d['image'].shape[0] == 3 for d in datapoint)
                for d in datapoint:
                    d['image'] = d['image'].flip(dims=(2,))
            image_filenames = [d['file_name'] for d in datapoint]
            input_images = [d['image'] for d in datapoint]
            input_images = [np.asarray(img.permute(1, 2, 0)[:, :, [2, 1, 0]]) for img in input_images]
            input_images_from_files = [cv2.imread(fn) for fn in image_filenames]
            input_images = [convert_datapoint_to_image_format(im, im2.shape[:2], cfg)
                            for im, im2 in zip(input_images, input_images_from_files)]

            for predictor_type, predictor in predictors.items():
                # n_existing_exporter_images = len(exporter_.generated_figures)
                cfg_tag = ('_flip' if rl else '') + f"_{predictor_type}"

                n_existing_exporter_images = len(exporter_.generated_figures)
                outputs_d = run_inference(predictor, datapoint)
                run_evaluation(input_images, cfg, outputs_d,
                               image_ids=[str(d['image_id']) + cfg_tag for d in datapoint],
                               model=predictor.model, exporter=exporter)
                my_image_ids = [str(d['image_id']) + cfg_tag for d in datapoint]
                for my_image_id in my_image_ids:
                    figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + my_image_id + '_collated'
                    collate_figures(exporter_.generated_figures[n_existing_exporter_images:], figure_name,
                                    exporter=exporter)

#
# def main(config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
#          image_ids=('486536',), flip_lr=False, exporter=exporter_):
#     if type(image_ids) is str:
#         image_ids = [image_ids]
#     cfg = get_maskrcnn_cfg()
#     custom_cfg = get_custom_maskrcnn_cfg()
#     predictor = DefaultPredictor(custom_cfg)
#     assert isinstance(predictor.model.roi_heads, MultiROIHeadsAPD)
#
#     rls = [False, True] if flip_lr is True else [False]
#     for rl in rls:
#         for image_id in image_ids:
#             saved_input_file = get_datapoint_file(cfg, image_id)
#             datapoint = torch.load(saved_input_file)
#             if type(datapoint) is not list:
#                 datapoint = [datapoint]
#             if rl:
#                 assert all(d['image'].shape[0] == 3 for d in datapoint)
#                 for d in datapoint:
#                     d['image'] = d['image'].flip(dims=(2,))
#             image_filenames = [d['file_name'] for d in datapoint]
#             input_images = [d['image'] for d in datapoint]
#             input_images = [np.asarray(img.permute(1, 2, 0)[:, :, [2, 1, 0]]) for img in input_images]
#             input_images_from_files = [cv2.imread(fn) for fn in image_filenames]
#             input_images = [convert_datapoint_to_image_format(im, im2.shape[:2], cfg)
#                             for im, im2 in zip(input_images, input_images_from_files)]
#
#             for predictor_type in MASK_HEAD_TYPES:
#                 # n_existing_exporter_images = len(exporter_.generated_figures)
#                 predictor.model.roi_heads
#                 cfg_tag = ('_flip' if rl else '') + f"_{predictor_type}"
#
#                 n_existing_exporter_images = len(exporter_.generated_figures)
#                 outputs_d = run_inference(predictors['standard'], datapoint)
#                 run_evaluation(input_images, cfg, outputs_d,
#                                image_ids=[str(d['image_id']) + cfg_tag for d in datapoint],
#                                model=predictors[predictor_type].model, exporter=exporter)
#                 my_image_ids = [str(d['image_id']) + cfg_tag for d in datapoint]
#                 for my_image_id in my_image_ids:
#                     figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + my_image_id + '_collated'
#                     collate_figures(exporter_.generated_figures[n_existing_exporter_images:], figure_name,
#                                     exporter=exporter)


if __name__ == '__main__':
    exporter_ = FigExporter()
    image_ids_ = ['486536', '306284', '9']
    main(image_ids=image_ids_, flip_lr=True, exporter=exporter_)
