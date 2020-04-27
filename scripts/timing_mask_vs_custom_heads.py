"""
The goal of train_two_masks_one_image is to overfit one image to two masks per box.  For the two cats - one chair
image, this means each cat box predicts two cat masks, and the chair box predicts on chair.

Step 1a: Modify multimask loss to send two different sets of weights / ground truth to the loss function
Step 1b: Verify with multimask inference to see that the set of weights is learning the primary or secondary mask,
depending on which one we switch between.

"""

import os
import torch
import torch.distributed
import torch.distributed

from maskrcnnextension.train import script_utils
from maskrcnnextension.analysis import vis_utils
from detectron2.utils.events import EventStorage
from maskrcnnextension.train.script_utils import run_inference, visualize_instancewise_predictions, prep_image
from maskrcnnextension.train.trainer_apd import Trainer_APD


VISUALIZE = 0

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def train_on_single_image(trainer: Trainer_APD, datapoint, max_itr=100, start_itr=0):
    assert max_itr >= start_itr
    with EventStorage(0) as trainer.storage:
        for t in range(start_itr, max_itr):
            if (t - start_itr) % 10 == 0:
                print(t, '/', max_itr)
            trainer.run_step_with_given_data(datapoint)


def stochastic_train_on_set(trainer: Trainer_APD, batches, max_itr=100, start_itr=0):
    N = len(batches)
    assert max_itr >= start_itr
    with EventStorage(0) as trainer.storage:
        for t in range(start_itr, max_itr):
            if (t - start_itr) % 10 == 0:
                print(t, '/', max_itr)
            print('Image {}'.format(t % N))
            trainer.run_step_with_given_data(batches[int(t % N)])


def stochastic_train_on_set2(trainer: Trainer_APD, batches, max_itr=100, start_itr=0):
    stochastic_train_on_set(trainer, batches, max_itr, start_itr)


def stochastic_train_on_set1(trainer: Trainer_APD, batches, max_itr=100, start_itr=0):
    stochastic_train_on_set(trainer, batches, max_itr, start_itr)


def main_custom(image_ids=('306284', '486536', '9'), maxitr=100, step=50):
    exporter = vis_utils.FigExporter()

    print('Beginning setup...')
    cfg = script_utils.get_custom_maskrcnn_cfg()
    trainer = Trainer_APD(cfg)

    print('Completing setup...')
    batches = []
    for image_id in image_ids:
        datapoint = torch.load(script_utils.get_datapoint_file(cfg, image_id))
        batches.append([datapoint] if type(datapoint) is not list else datapoint)

    # head_type = 'custom' if custom_head else 'standard'
    head_type = 'custom'
    script_utils.activate_head_type(trainer, head_type)
    cfg_tag = f"_{head_type}"

    if VISUALIZE:
        for image_id, batch in zip(image_ids, batches):
            for masks_key in ['pred_masks1', 'pred_masks2']:
                pretraining(batch, cfg, cfg_tag + '_' + masks_key.replace('pred_masks', 'm'), exporter, image_id, trainer,
                            masks_key=masks_key)

    input_images = [prep_image(batch[0], cfg) for batch in batches]
    strt = 0
    for strtitr in range(0, maxitr, step):
        itr = strtitr + step
        stochastic_train_on_set1(trainer, batches, max_itr=strtitr + itr, start_itr=strtitr)
        # visualize

        if VISUALIZE:
            for masks_key in ['pred_masks1', 'pred_masks2']:
                for image_id, input_image, batch in zip(image_ids, input_images, batches):
                    posttraining(cfg, cfg_tag + '_' + masks_key.replace('pred_masks', 'm'), batch[0], exporter, image_id,
                                 input_image, itr, trainer, masks_key=masks_key, show_pipeline=False)


def main_standard(image_ids=('306284', '486536', '9'), maxitr=100, step=50):
    exporter = vis_utils.FigExporter()

    print('Beginning setup...')
    cfg = script_utils.get_custom_maskrcnn_cfg()
    trainer = Trainer_APD(cfg)

    print('Completing setup...')
    batches = []
    for image_id in image_ids:
        datapoint = torch.load(script_utils.get_datapoint_file(cfg, image_id))
        batches.append([datapoint] if type(datapoint) is not list else datapoint)

    # head_type = 'custom' if custom_head else 'standard'
    head_type = 'standard'
    script_utils.activate_head_type(trainer, head_type)

    cfg_tag = f"_{head_type}"

    if VISUALIZE:
        for image_id, batch in zip(image_ids, batches):
            for masks_key in ['pred_masks']:
                pretraining(batch, cfg, cfg_tag + '_' + masks_key.replace('pred_masks', 'm'), exporter, image_id, trainer,
                            masks_key=masks_key)

    input_images = [prep_image(batch[0], cfg) for batch in batches]
    strt = 0
    for strtitr in range(0, maxitr, step):
        itr = strtitr + step
        stochastic_train_on_set2(trainer, batches, max_itr=strtitr + itr, start_itr=strtitr)
        # visualize

        if VISUALIZE:
            for masks_key in ['pred_masks']:
                for image_id, input_image, batch in zip(image_ids, input_images, batches):
                    posttraining(cfg, cfg_tag + '_' + masks_key.replace('pred_masks', 'm'), batch[0], exporter, image_id,
                                 input_image, itr, trainer, masks_key=masks_key, show_pipeline=False)


def main_orig(image_ids=('306284', '486536', '9'), maxitr=1000, step=500):
    exporter = vis_utils.FigExporter()

    print('Beginning setup...')
    cfg = script_utils.get_maskrcnn_cfg()
    trainer = Trainer_APD(cfg)

    print('Completing setup...')
    batches = []
    for image_id in image_ids:
        datapoint = torch.load(script_utils.get_datapoint_file(cfg, image_id))
        batches.append([datapoint] if type(datapoint) is not list else datapoint)

    # head_type = 'custom' if custom_head else 'standard'
    head_type = 'standard'
    script_utils.activate_head_type(trainer, head_type)
    cfg_tag = f"_{head_type}"

    if VISUALIZE:
        for image_id, batch in zip(image_ids, batches):
            for masks_key in ['pred_masks']:
                pretraining(batch, cfg, cfg_tag + '_' + masks_key.replace('pred_masks', 'm'), exporter, image_id, trainer,
                            masks_key=masks_key)

    input_images = [prep_image(batch[0], cfg) for batch in batches]
    strt = 0
    for strtitr in range(0, maxitr, step):
        itr = strtitr + step
        stochastic_train_on_set(trainer, batches, max_itr=strtitr + itr, start_itr=strtitr)
        # visualize

        if VISUALIZE:
            for masks_key in ['pred_masks']:
                for image_id, input_image, batch in zip(image_ids, input_images, batches):
                    posttraining(cfg, cfg_tag + '_' + masks_key.replace('pred_masks', 'm'), batch[0], exporter, image_id,
                                 input_image, itr, trainer, masks_key=masks_key, show_pipeline=False)


def posttraining(cfg, cfg_tag, dpt, exporter, image_id, input_image, max_iters, trainer, masks_key='pred_masks',
                 show_pipeline=False):
    outputs_d = run_inference(trainer, [dpt])
    if masks_key is None:
        pass
    else:
        assert masks_key.startswith('pred_masks')
        for o in outputs_d['outputs']:
            o['instances'].set('pred_masks', o['instances'].get(masks_key))
    outputs = outputs_d['outputs'][0]

    img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                 pred_instances=outputs['instances'],
                                                                 proposals=None)
    posttrain_tag = f'_posttrain_{max_iters}'
    visualize_instancewise_predictions(
        img, pred_instances, cfg, exporter, tag=f'{image_id}' + posttrain_tag + cfg_tag)
    figure_name = image_id + cfg_tag + posttrain_tag + '_collated'
    figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + figure_name,
                                        delete_individuals=True)
    dbprint('Exported', figname)
    if show_pipeline:
        script_utils.run_single_image_results_visualization(cfg, exporter, outputs_d['extra_proposal_details'][0],
                                                            image_id, img, trainer.model, outputs, outputs_d[
                                                                'proposalss'][0],
                                                            visualize_just_image=False)
        figname = exporter.collate_previous(
            '{:02d}'.format(exporter.fig_number) + 'c_' + os.path.splitext(os.path.basename(__file__))[
                0] + '_' + image_id + cfg_tag +
            posttrain_tag + '_pipeline', delete_individuals=True)
        dbprint('Exported', figname)


def pretraining(batch, cfg, cfg_tag, exporter, image_id, trainer, masks_key=None):
    outputs_d = run_inference(trainer, batch)
    if masks_key is None:
        pass
    else:
        assert masks_key.startswith('pred_masks')
        for o in outputs_d['outputs']:
            o['instances'].set('pred_masks', o['instances'].get(masks_key))
            o['instances'].pred_masks = o['instances'].get(masks_key)
            assert torch.all(o['instances'].pred_masks == o['instances'].get(masks_key))

    for dpt, outputs in zip(batch, outputs_d['outputs']):
        vis_utils.visualize_instancewise_groundtruth(dpt, cfg, exporter, cfg_tag + '_gt')
        exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + image_id + '_gt',
                                  delete_individuals=True)

        input_image = prep_image(dpt, cfg)
        img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                     pred_instances=outputs['instances'],
                                                                     proposals=None)
        posttrain_tag = '_pretrain'
        visualize_instancewise_predictions(img, pred_instances, cfg, exporter, tag=f'{image_id}' + posttrain_tag +
                                                                                   cfg_tag)
        figure_name = os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + cfg_tag + \
                      posttrain_tag + '_collated'
        figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + figure_name,
                                            delete_individuals=True)
        dbprint('Exported', figname)
        script_utils.run_single_image_results_visualization(cfg, exporter, outputs_d['extra_proposal_details'][0],
                                                            image_id, img, trainer.model, outputs, outputs_d[
                                                                'proposalss'][0],
                                                            visualize_just_image=False)
        figname = exporter.collate_previous(
            '{:02d}'.format(exporter.fig_number) + 'c_' + os.path.splitext(os.path.basename(__file__))[
                0] + '_' + image_id + cfg_tag +
            posttrain_tag + '_pipeline', delete_individuals=True)
    return outputs_d


def main(maxitr=1000, step=500):
    main_standard(maxitr=maxitr, step=step)
    main_custom(maxitr=maxitr, step=step)


if __name__ == '__main__':
    main()
