"""
The goal of single_image_overfit_one_proposal is to prove we can overfit to a single selected proposal.  Reasoning:
- With the customized head, we need to specialize the loss with respect to each proposal
- With the customized head, we may be using only a subset of the proposals, or a different set entirely (interfacing
at an earlier level).  By fitting to a single proposal, we're demonstrating that we are able to interface at the
proposal rather than image level.
- Our next step will be to change this loss to the customized loss.
"""

import cv2
import numpy as np
import os
import torch
import torch.distributed

from detectron2.data import MetadataCatalog
from maskrcnnextension.analysis import vis_utils
from maskrcnnextension.train import script_utils
from detectron2.utils.events import EventStorage
from maskrcnnextension.train.script_utils import get_custom_maskrcnn_cfg, get_datapoint_file, \
    convert_datapoint_to_image_format, run_inference, visualize_instancewise_predictions
from maskrcnnextension.analysis.vis_utils import FigExporter
from maskrcnnextension.train.trainer_apd import Trainer_APD
from detectron2.modeling.roi_heads.multi_mask_head_apd import CustomMaskRCNNConvUpsampleHeadAPD

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def equal_ids(id1, id2):
    return str(id1).rstrip('0') == str(id2).rstrip('0')


def prep_image(datapoint, cfg):
    image_filename = datapoint['file_name']
    input_image = datapoint['image']
    input_image = np.asarray(input_image.permute(1, 2, 0)[:, :, [2, 1, 0]])
    input_image_from_file = cv2.imread(image_filename)
    input_image = convert_datapoint_to_image_format(input_image, input_image_from_file.shape[:2], cfg)
    return input_image


def get_formatted_boxes_matched_to_predictions(predictor, batch):
    out_d = run_inference(predictor, batch)
    assert len(batch) == 1
    bi = 0  # index into batch
    # 'extra_proposal_details' contains 'selected_proposal_idxs', 'boxes' (1000,4*80), and 'scores' (1000, 81)
    selected_proposal_idxs = out_d['extra_proposal_details']['selected_proposal_idxs'][bi]
    chosen_proposals = out_d['proposalss'][bi][selected_proposal_idxs]
    predictions = out_d['outputs'][bi]['instances']
    chosen_proposals.pred_classes = predictions.pred_classes
    chosen_proposals.pred_boxes = chosen_proposals.proposal_boxes
    return chosen_proposals, predictions


def train_on_single_image(trainer: Trainer_APD, datapoint, max_itr=100, start_itr=0):
    assert max_itr >= start_itr
    with EventStorage(0) as trainer.storage:
        for t in range(max_itr - start_itr):
            trainer.run_step_with_given_data(datapoint)


def activate_head_type(trainer, head_type):
    trainer.model.roi_heads.active_mask_head = head_type
    if head_type is 'custom':
        assert type(trainer.model.roi_heads.mask_heads[trainer.model.roi_heads.active_mask_head]) is \
               CustomMaskRCNNConvUpsampleHeadAPD, 'Not using custom head; head type is {}'.format(type(
            trainer.model.roi_heads.mask_heads[trainer.model.roi_heads.active_mask_head]))
    else:
        assert type(trainer.model.roi_heads.mask_heads[trainer.model.roi_heads.active_mask_head]) \
               is not CustomMaskRCNNConvUpsampleHeadAPD


def main(image_id='486536', max_iters=(1,), exporter: vis_utils.FigExporter = None, head_type='custom'):
    exporter = exporter or FigExporter()

    print('Beginning setup...')
    cfg = get_custom_maskrcnn_cfg()
    trainer = Trainer_APD(cfg)

    print('Completing setup...')
    standard_head = trainer.model.roi_heads.mask_heads['standard']
    custom_head = trainer.model.roi_heads.mask_heads['custom']
    if not torch.equal(standard_head.predictor.bias,
                       custom_head.predictor.bias[::custom_head.num_instances_per_class, ...]):
        raise Exception
    if not torch.equal(standard_head.predictor.weight,
                       custom_head.predictor.weight[::custom_head.num_instances_per_class, ...]):
        raise Exception

    datapoint = torch.load(get_datapoint_file(cfg, image_id))

    if type(datapoint) is list:
        datapoint = datapoint[0]
    batch = [datapoint]

    # head_type = 'custom' if custom_head else 'standard'
    activate_head_type(trainer, head_type)
    cfg_tag = f"_{head_type}"
    print('Pretraining')
    custom_outputs_d = pretraining(batch, cfg, cfg_tag, exporter, image_id, trainer)
    if 0:
        head_type = 'standard'
        cfg_tag = f"_{head_type}"
        activate_head_type(trainer, head_type)
        standard_outputs_d = pretraining(batch, cfg, cfg_tag, exporter, image_id, trainer)
    cfg_tag = f"_{head_type}"
    n_itrs = 0

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    names = metadata.get("thing_classes", None)
    labels = [names[i] for i in datapoint['instances'].gt_classes]
    for maxitr in sorted(max_iters):
        for dpt in batch:
            # Train
            print('Training on single image with {} steps'.format(maxitr))
            train_on_single_image(trainer, batch, max_itr=maxitr, start_itr=n_itrs)

            # Check inference
            input_image = prep_image(dpt, cfg)
            posttraining(cfg, cfg_tag, dpt, exporter, image_id, input_image, maxitr, trainer)
            n_itrs += maxitr


def posttraining(cfg, cfg_tag, dpt, exporter, image_id, input_image, max_iters, trainer):
    outputs_d = run_inference(trainer, [dpt])
    outputs = outputs_d['outputs'][0]
    img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                 pred_instances=outputs['instances'],
                                                                 proposals=None)
    posttrain_tag = f'_posttrain_{max_iters}'
    visualize_instancewise_predictions(img, pred_instances, cfg, exporter, tag=f'{image_id}' + posttrain_tag +
                                                                               cfg_tag)
    figure_name = image_id + cfg_tag + posttrain_tag + '_collated'
    figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + figure_name,
                                        delete_individuals=True)
    dbprint('Exported', figname)
    script_utils.run_single_image_results_visualization(cfg, exporter, outputs_d['extra_proposal_details'][0],
                                                        image_id, img, trainer.model, outputs, outputs_d[
                                                            'proposalss'][0],
                                                        visualize_just_image=False)
    figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag +
                                        posttrain_tag + '_pipeline', delete_individuals=True)
    dbprint('Exported', figname)


def pretraining(batch, cfg, cfg_tag, exporter, image_id, trainer):
    outputs_d = run_inference(trainer, batch)
    for dpt, outputs in zip(batch, outputs_d['outputs']):
        vis_utils.visualize_instancewise_groundtruth(dpt, cfg, exporter, cfg_tag + '_gt')
        exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + image_id + '_gt', delete_individuals=True)
        input_image = prep_image(dpt, cfg)
        img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                     pred_instances=outputs['instances'],
                                                                     proposals=None)
        posttrain_tag = '_pretrain'
        visualize_instancewise_predictions(img, pred_instances, cfg, exporter, tag=f'{image_id}' + posttrain_tag +
                                                                                   cfg_tag)
        figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag + \
                      posttrain_tag + '_collated'
        figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + figure_name, delete_individuals=True)
        dbprint('Exported', figname)
        script_utils.run_single_image_results_visualization(cfg, exporter, outputs_d['extra_proposal_details'][0],
                                                            image_id, img, trainer.model, outputs, outputs_d[
                                                                'proposalss'][0],
                                                            visualize_just_image=False)
        figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag +
                                            posttrain_tag + '_pipeline', delete_individuals=True)
    return outputs_d


if __name__ == '__main__':
    exporter_ = FigExporter()
    image_ids_ = ['486536', '306284', '9']
    for head_type in ['custom', 'standard']:
        main(image_id=image_ids_[0], max_iters=[10, 100], exporter=exporter_, head_type=head_type)
