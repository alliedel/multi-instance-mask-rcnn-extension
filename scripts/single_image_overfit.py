"""
The goal of single_image_overfit is to prove we can overfit to a single image.  Reasoning:
- With the customized head, we need to specialize the loss with respect to each proposal
- With the customized head, we may be using only a subset of the proposals, or a different set entirely (interfacing
at an earlier level).  By fitting to a single proposal, we're demonstrating that we are able to interface at the
proposal rather than image level.
- Our next step will be to change this loss to the customized loss.
"""

import os
import torch
import torch.distributed

from detectron2.data import MetadataCatalog
from multimaskextension.analysis import vis_utils
from multimaskextension.train import script_utils
from detectron2.utils.events import EventStorage
from multimaskextension.train.script_utils import get_custom_maskrcnn_cfg, run_batch_results_visualization, get_datapoint_file, \
    prep_image, run_inference
from multimaskextension.train.trainer_apd import Trainer_APD
from multimaskextension.analysis.vis_utils import visualize_groundtruth, FigExporter

exporter_ = None


def train_on_single_image(trainer: Trainer_APD, batch, max_iters=100):
    for t in range(max_iters):
        with EventStorage(0) as trainer.storage:
            trainer.run_step_with_given_data(batch)


def visualize_single_instance_predictions(img, instance_outputs, cfg, exporter, tag):
    n_instances = len(instance_outputs)
    for i in range(n_instances):
        instance = instance_outputs[[i]]
        vis_utils.show_prediction(img, {'instances': instance}, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        exporter.export_gcf(tag + f'_inst{i}')


def main(image_id='486536', custom_head=True, max_iters=1, exporter=None):
    exporter = exporter or FigExporter()
    cfg = get_custom_maskrcnn_cfg()
    trainer = Trainer_APD(cfg)
    datapoint = torch.load(get_datapoint_file(cfg, image_id))

    if type(datapoint) is list:
        datapoint = datapoint[0]

    visualize_groundtruth(cfg, datapoint, exporter, image_id)

    input_image = prep_image(datapoint, cfg)
    trainer_type = 'custom' if custom_head else 'standard'
    trainer.model.roi_heads.active_mask_head = trainer_type
    cfg_tag = f"_{trainer_type}"
    n_existing_exporter_images = len(exporter.generated_figures)

    batch = [datapoint]

    # Pre-training inference
    outputs_d = run_inference(trainer, batch)
    for dpt, outputs in zip(batch, outputs_d['outputs']):
        input_image = prep_image(dpt, cfg)
        img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                     pred_instances=outputs['instances'],
                                                                     proposals=None)
        posttrain_tag = '_pretrain'
        n_existing_exporter_images = len(exporter.generated_figures)
        visualize_single_instance_predictions(img, pred_instances, cfg, exporter, tag=f'{image_id}' + posttrain_tag +
                                                                                      cfg_tag)
        figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag + \
                      posttrain_tag + '_collated'
        figname = vis_utils.collate_figures(exporter.generated_figures[n_existing_exporter_images:], figure_name,
                                            exporter=exporter, delete_individuals=True)
        n_existing_exporter_images = len(exporter.generated_figures)

        # figure_name = visualize_results(batch, cfg, cfg_tag, exporter, image_id, input_image,
        #                                 n_existing_exporter_images, outputs_d, '_pretrain', trainer)
        # n_existing_exporter_images = len(exporter.generated_figures)

        # Train
        train_on_single_image(trainer, batch, max_iters=max_iters)

        # Check inference
        outputs = run_inference(trainer, [dpt])['outputs'][0]
        img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                     pred_instances=outputs['instances'],
                                                                     proposals=None)
        posttrain_tag = f'_posttrain_{max_iters}'
        # visualize_results(batch, cfg, cfg_tag, exporter, image_id, input_image, n_existing_exporter_images, outputs_d,
        #                   posttrain_tag, trainer)
        visualize_single_instance_predictions(img, pred_instances, cfg, exporter, tag=f'{image_id}' + posttrain_tag +
                                                                                      cfg_tag)
        figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag + \
                      posttrain_tag + '_collated'
        figname = vis_utils.collate_figures(exporter.generated_figures[n_existing_exporter_images:], figure_name,
                                            exporter=exporter, delete_individuals=True)


        n_existing_exporter_images = len(exporter.generated_figures)


def visualize_results(batch, cfg, cfg_tag, exporter, image_id, input_image, n_existing_exporter_images, outputs_d,
                      posttrain_tag, trainer):
    # create artificial output_d's so that each proposal is visualized one at a time.
    run_batch_results_visualization([input_image], cfg, outputs_d,
                                    image_ids=[str(d['image_id']) + cfg_tag + posttrain_tag for d in batch],
                                    model=trainer.model, exporter=exporter)
    figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag + \
                  posttrain_tag + '_collated'
    figname = vis_utils.collate_figures(exporter.generated_figures[n_existing_exporter_images:], figure_name,
                                        exporter=exporter, delete_individuals=True)
    for dpt, outputs in zip(batch, outputs_d['outputs']):
        input_image = prep_image(dpt, cfg)
        img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                     pred_instances=outputs['instances'],
                                                                     proposals=None)
        visualize_single_instance_predictions(img, pred_instances, cfg, exporter, tag=f'{image_id}' + '_posttrain' +
                                                                                      cfg_tag)

    return figname


if __name__ == '__main__':
    exporter_ = FigExporter()
    image_ids_ = ['486536', '306284', '9']
    for max_iters_ in [10, 100]:  # [1, 100]
        main(image_id=image_ids_[0], max_iters=max_iters_, exporter=exporter_)
