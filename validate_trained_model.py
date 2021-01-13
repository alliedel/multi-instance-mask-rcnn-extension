"""
The goal of train_two_masks_one_image is to overfit one image to two masks per box.  For the two cats - one chair
image, this means each cat box predicts two cat masks, and the chair box predicts on chair.

Step 1a: Modify multimask loss to send two different sets of weights / ground truth to the loss function
Step 1b: Verify with multimask inference to see that the set of weights is learning the primary or secondary mask,
depending on which one we switch between.

"""
import argparse
import gc

import local_pyutils
import os
import torch
import torch.distributed
import torch.distributed

from detectron2.utils.events import EventStorage
from multimaskextension.analysis import vis_utils
from multimaskextension.train import script_utils
from multimaskextension.train.script_utils import run_inference, visualize_instancewise_predictions, prep_image
from multimaskextension.train.trainer_apd import Trainer_APD

# We do this to force the multiroiheads to be added to the registry
from multimaskextension.model import multi_roi_heads_apd
from multimaskextension.data import registryextension


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

# COCO: image_ids=('306284', '486536', '9')
def main(resume, cfg_file, image_ids=None):
    exporter = vis_utils.FigExporter()

    print('Beginning setup...')
    assert os.path.exists(cfg_file), cfg_file
    cfg = script_utils.get_custom_maskrcnn_cfg(cfg_file)
    trainer = Trainer_APD(cfg, out_dir='~/workspace', checkpoint_resume=resume)
    head_type = 'custom'
    script_utils.activate_head_type(trainer, head_type)

    print('Completing setup...')
    batches = []
    if image_ids is not None:
        for image_id in image_ids:
            datapoint = torch.load(script_utils.get_datapoint_file(cfg, image_id))
            batches.append([datapoint] if type(datapoint) is not list else datapoint)
    else:
        cachedir = './output/cache/input/'
        assert os.path.isdir(cachedir)
        # predictor.model.training = True
        dataloader = script_utils.build_dataloader(cfg)
        n_batches = 2
        image_ids = []
        for batch_i, ds in enumerate(dataloader):
            if batch_i >= n_batches:
                break
            for d in ds:
                image_id = d['image_id']
                image_ids.append(image_id)
                saved_input_file = os.path.join(cachedir, f"input_{image_id}.pt")
                if not os.path.exists(saved_input_file):
                    datapoint = d
                    torch.save(datapoint, saved_input_file)
        print(image_ids)
        gc.collect()
        del dataloader
        for image_id in image_ids:
            datapoint = torch.load(script_utils.get_datapoint_file(cfg, image_id))
            batches.append([datapoint] if type(datapoint) is not list else datapoint)

    input_images = [prep_image(batch[0], cfg) for batch in batches]

    cfg_tag = 'cfg_tag'
    itr = trainer.iter
    for masks_key in ['pred_masks1', 'pred_masks2']:
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
            try:
                o['instances'].set('pred_masks_soft', o['instances'].get(masks_key + '_soft'))
            except AttributeError:
                print(Warning('Soft scores not found: {}'.format(masks_key + '_soft')))
            o['instances'].set('pred_masks', o['instances'].get(masks_key))
    outputs = outputs_d['outputs'][0]

    img, pred_instances, _ = script_utils.prep_for_visualization(cfg, input_image,
                                                                 pred_instances=outputs['instances'],
                                                                 proposals=None)
    posttrain_tag = f'_posttrain_{max_iters}'
    visualize_instancewise_predictions(
        img, pred_instances, cfg, exporter, tag=f'{image_id}' + posttrain_tag + cfg_tag)
    figure_name = f"{image_id}{cfg_tag}{posttrain_tag}_pred"
    figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + figure_name,
                                        delete_individuals=True)
    dbprint('Exported', figname)
    vis_utils.visualize_instancewise_soft_predictions(
        pred_instances, exporter, tag=f'{image_id}' + posttrain_tag + cfg_tag)
    figure_name = f"{image_id}{cfg_tag}{posttrain_tag}_pred_soft"
    figname = exporter.collate_previous('{:02d}'.format(exporter.fig_number) + 'c_' + figure_name,
                                        delete_individuals=True)
    dbprint('Exported', figname)
    if show_pipeline:
        script_utils.run_single_image_results_visualization(cfg, exporter, outputs_d['extra_proposal_details'][0],
                                                            image_id, img, trainer.model, outputs, outputs_d[
                                                                'proposalss'][0],
                                                            visualize_just_image=False)
        nm = f"{'{:02d}'.format(exporter.fig_number)}c_{os.path.splitext(os.path.basename(__file__))[0]}" \
             f"_{image_id}{cfg_tag}{posttrain_tag}_pipeline"
        figname = exporter.collate_previous(nm, delete_individuals=True)
        dbprint('Exported', figname)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', required=False, default=None)
    parser.add_argument('--cfg-pth', required=False, default=None)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(resume=args.model_pth, cfg_file=args.cfg_pth)
