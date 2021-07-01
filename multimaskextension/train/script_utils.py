import pickle
import subprocess

import copy
import cv2
import fvcore.nn.weight_init as weight_init
import gc
import logging
import numpy as np
import os
import time
import torch
import torch.distributed
from PIL import Image
from torch import nn

from multimaskextension.analysis import vis_utils
from multimaskextension.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation.evaluator import inference_context
from detectron2.modeling import detector_postprocess
from multimaskextension.model.multi_mask_head_apd import CustomMaskRCNNConvUpsampleHeadAPD
from multimaskextension.analysis.vis_utils import visualize_single_image_output, input_img_to_rgb

DETECTRON_REPO = './detectron2_repo'
if os.path.isdir('data/models'):
    DETECTRON_MODEL_ZOO = 'data/models/detectron_model_zoo'
else:
    if 'DATAPATH' not in os.environ:
        assert 'Could not find data/models.  Symlink to ./data, ' \
               'or set the DATAPATH environment variable'
    DETECTRON_MODEL_ZOO = os.path.join(os.environ['DATAPATH'], 'models/detectron_model_zoo')
assert os.path.isdir(DETECTRON_MODEL_ZOO), DETECTRON_MODEL_ZOO


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def make_multihead_state_from_single_head_state(existing_weights_filename,
                                                new_weights_filename,
                                                src_parent_name='roi_heads.mask_head',
                                                dst_parent_names=('roi_heads.standard_mask_head',
                                                                  'roi_heads.custom_mask_head'),
                                                remove_src_keys=True):
    if existing_weights_filename.endswith(".pkl"):
        with open(existing_weights_filename, "rb") as f:
            existing_state = pickle.load(f, encoding="latin1")
    else:
        existing_state = torch.load(existing_weights_filename)
    extra_state = {}
    src_keys_copied = []
    src_found = False
    for src_key in existing_state['model']:
        if src_key.startswith(src_parent_name):
            src_found = True
            src_keys_copied.append(src_key)
            for dst_parent_name in dst_parent_names:
                extra_state[src_key.replace(src_parent_name, dst_parent_name)] = \
                    copy.deepcopy(existing_state['model'][src_key])
    if not src_found:
        raise Exception(f"{src_parent_name} not found in {existing_weights_filename}")
    if remove_src_keys:
        for src_key in src_keys_copied:
            existing_state['model'].pop(src_key)  # okay if src_key in dst_parent_names: will be updated next.
    existing_state['model'].update(extra_state)
    if existing_weights_filename.endswith(".pkl"):
        pickle.dump(existing_state, open(new_weights_filename, 'wb'))
    else:
        torch.save(existing_state, new_weights_filename)


def initialize_to_random(mask_head):
    for layer in mask_head.conv_norm_relus + [mask_head.deconv]:
        weight_init.c2_msra_fill(layer)
    # use normal distribution initialization for mask prediction layer
    nn.init.normal_(mask_head.predictor.weight, std=0.001)
    if mask_head.predictor.bias is not None:
        nn.init.constant_(mask_head.predictor.bias, 0)


def copy_singlemask_multihead_to_multimask_multihead(existing_weights_filename,
                                                     new_weights_filename, n_masks_per_roi=2,
                                                     src_parent_name='roi_heads.custom_mask_head.predictor',
                                                     dst_parent_name='roi_heads.custom_mask_head.predictor',
                                                     remove_src_keys=True):
    """
    Standard Mask R-CNN predicts one mask per ROI.  This creates a state dict that has random weights for any
    additional masks produced per ROI.  Note the copied and random weights are 'interleaved' because we masks of the
    same class are adjacent in channels (dog1, dog2, cat1, cat2): NOT (dog1, cat1, dog2, cat2)
    """

    if existing_weights_filename.endswith(".pkl"):
        with open(existing_weights_filename, "rb") as f:
            existing_state = pickle.load(f, encoding="latin1")
    else:
        existing_state = torch.load(existing_weights_filename)
    extra_state = {}
    src_keys_copied = []
    src_found = False
    for src_key in existing_state['model']:
        if src_key.startswith(src_parent_name):
            src_found = True
            src_keys_copied.append(src_key)
            n_classes = existing_state['model'][src_key].shape[0]  # C, 256, 1, 1
            dst_key = src_key.replace(src_parent_name, dst_parent_name)
            if src_key.endswith('weight'):
                extra_state[dst_key] = repeat_interleave(existing_state['model'][src_key], n_masks_per_roi, dim=0)
                for roi_m in range(n_masks_per_roi):
                    if roi_m == 0:
                        assert np.array_equal(extra_state[dst_key][roi_m::n_masks_per_roi, :, :, :],
                                              existing_state['model'][src_key])
                    else:
                        # use normal distribution initialization for mask prediction layer
                        tensor_vsn = torch.Tensor(extra_state[dst_key][roi_m::n_masks_per_roi, :, :, :])
                        nn.init.normal_(tensor_vsn, std=0.001)
                        extra_state[dst_key][roi_m::n_masks_per_roi, :, :, :] = \
                            tensor_vsn.numpy().astype(dtype=extra_state[dst_key].dtype)
            elif src_key.endswith('bias'):
                if existing_state['model'][src_key] is None:
                    extra_state[dst_key] = None
                else:
                    extra_state[dst_key] = repeat_interleave(existing_state['model'][src_key], n_masks_per_roi, dim=0)
                    copy.deepcopy(existing_state['model'][src_key])
                    for roi_m in range(n_masks_per_roi):
                        if roi_m == 0:
                            assert np.array_equal(extra_state[dst_key][roi_m::n_masks_per_roi],
                                                  existing_state['model'][src_key])
                        else:
                            # nn.init.constant_(extra_state[dst_key][roi_m::n_masks_per_roi], 0)
                            extra_state[dst_key][roi_m::n_masks_per_roi] = 0
            else:
                raise NotImplementedError('Dont know how to copy {}'.format(src_key))
    if not src_found:
        raise Exception(f"{src_parent_name} not found in {existing_weights_filename}")
    if remove_src_keys:
        for src_key in src_keys_copied:
            if src_key in existing_state['model']:
                existing_state['model'].pop(src_key)
    existing_state['model'].update(extra_state)

    if not np.array_equal(existing_state['model']['roi_heads.standard_mask_head.predictor.weight'],
                          existing_state['model']['roi_heads.custom_mask_head.predictor.weight'][::n_masks_per_roi,
                          ...]):
        raise Exception
    if not np.array_equal(existing_state['model']['roi_heads.standard_mask_head.predictor.bias'],
                          existing_state['model']['roi_heads.custom_mask_head.predictor.bias'][
                          ::n_masks_per_roi, ...]):
        raise Exception

    if existing_weights_filename.endswith(".pkl"):
        pickle.dump(existing_state, open(new_weights_filename, 'wb'))
    else:
        torch.save(existing_state, new_weights_filename)


def repeat_interleave(arr, n_repeats, dim=0):
    if torch.is_tensor(arr):
        return arr.repeat_interleave(n_repeats, dim=dim)
    else:
        return np.repeat(arr, n_repeats, axis=dim)


def download_detectron_model_to_local_zoo(relpath):
    if relpath.startswith('detectron2://'):
        relpath.replace('detectron2://', '', 1)
    url = 'https://dl.fbaipublicfiles.com/detectron2/' + relpath
    outpath = os.path.join(DETECTRON_MODEL_ZOO, relpath)
    outdir = os.path.dirname(outpath)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    assert os.path.isdir(outdir)
    if os.path.exists(outpath):
        return outpath
    try:
        stdout, stderr = subprocess.check_call(['wget', url, '-O', outpath])
    except subprocess.CalledProcessError:
        # print(stderr)
        raise
    return outpath


def get_custom_maskrcnn_cfg(config_filepath=f"configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_APD.yaml",
                            weights_checkpoint=None):
    cfg = get_cfg()
    assert os.path.exists(config_filepath), f"{config_filepath} does not exist"
    cfg.merge_from_file(config_filepath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url,
    # or use the
    # following shorthand
    # Adjust state dict for multiple heads
    if weights_checkpoint is None:  # only do this if running initialization (from a standard head file)
        model_rel_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
        local_path = download_detectron_model_to_local_zoo(model_rel_path)
        cfg.MODEL.WEIGHTS = local_path
        standard_state_file = cfg.MODEL.WEIGHTS
        ext = os.path.splitext(standard_state_file)[1]
        if cfg.MODEL.ROI_HEADS.NAME == "MultiROIHeadsAPD":
            multihead_state_file = standard_state_file.replace(f'{ext}', f'_multiheads_apd{ext}')
            if not os.path.exists(multihead_state_file):
                make_multihead_state_from_single_head_state(standard_state_file, multihead_state_file)
            custom_state_file = multihead_state_file
        else:
            custom_state_file = standard_state_file

        # Adjust state dict for multiple masks
        if cfg.MODEL.ROI_MASK_HEAD.N_MASKS_PER_ROI != 1:
            multimask_state_file = custom_state_file.replace(f'{ext}', f'_multiheads_apd{ext}')
            if not os.path.exists(multimask_state_file):
                copy_singlemask_multihead_to_multimask_multihead(custom_state_file, multimask_state_file)
            custom_state_file = multimask_state_file
    else:
        custom_state_file = weights_checkpoint
    cfg.MODEL.WEIGHTS = custom_state_file
    return cfg


def get_maskrcnn_cfg(
        config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    cfg = get_cfg()
    assert os.path.exists(config_filepath), f"{config_filepath} does not exist"
    cfg.merge_from_file(config_filepath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url,
    # or use the
    # following shorthand
    model_rel_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    local_path = download_detectron_model_to_local_zoo(model_rel_path)
    cfg.MODEL.WEIGHTS = local_path
    return cfg


def just_inference_on_dataset(model, data_loader, outdir, stop_after_n_points=None, get_proposals=False):
    """
    Function by Allie.

    Run model (in eval mode) on the data_loader and evaluate the metrics with evaluator.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    cuda = next(model.parameters()).is_cuda
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    n_points = len(data_loader)
    proposals_outdir = os.path.join(outdir, 'proposals') if get_proposals else None
    if get_proposals:
        if os.path.exists(proposals_outdir):
            raise Exception('Proposals outdir {} already exists.  Please delete.')
        os.makedirs(proposals_outdir)

    inference_outdir = os.path.join(outdir, 'predictions')
    if os.path.exists(inference_outdir):
        raise Exception(f"Predictions outdir {inference_outdir} already exists.  Please delete.")
    os.makedirs(inference_outdir)
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):

            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0
            elif stop_after_n_points is not None and idx >= stop_after_n_points:
                break

            start_compute_time = time.time()
            outputs = model(inputs)

            if get_proposals:
                images = model.preprocess_image(inputs)
                features = model.backbone(images.tensor)
                proposalss, proposal_lossess = model.proposal_generator(images, features, None)
            if cuda:
                torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            print(idx, '/', n_points)
            if data_loader.batch_sampler.batch_size != 1:
                raise NotImplementedError
            else:
                assert len(outputs) == 1
                assert len(inputs) == 1
            for output, input in zip(outputs, inputs):
                output.update({
                    k: input[k] for k in ('file_name', 'image_id')
                })
                torch.save(output, os.path.join(inference_outdir, 'output_' +
                                                os.path.splitext(os.path.basename(input['file_name']))[0] + '.pt'))
            if get_proposals:
                for proposals, input in zip(proposalss, inputs):
                    torch.save(proposals, os.path.join(outdir, 'proposals', 'proposals_' +
                                                       os.path.splitext(os.path.basename(input['file_name']))[
                                                           0] + '.pt'))
    return


def proposal_predictor_forward_pass(predictor, batched_inputs):
    """
    Instead of running the forward pass of the full R-CNN model, we extract the proposals.
    """

    images = predictor.model.preprocess_image(batched_inputs)
    features = predictor.model.backbone(images.tensor)
    proposals, proposal_losses = predictor.model.proposal_generator(images, features, None)
    return proposals


def get_image_identifiers(data_loader, identifier_strings=('file_name', 'image_id'), n_images_stop=None):
    assert data_loader.batch_sampler.batch_size == 1, NotImplementedError(
        'Only handing case of batch size = 1 for now')
    assert isinstance(data_loader.sampler, torch.utils.data.sampler.SequentialSampler), \
        'The data loader is not sequential, so the ordering will not be consistent if I give you the filenames.  ' \
        'Choose a data loader with a sequential sampler.'
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 50
    all_identifiers = []
    print('Collecting image identifiers')
    for idx, inputs in enumerate(data_loader):
        if idx % 10 == 0:
            print(idx, '/', n_images_stop or len(data_loader))
        x = inputs[0]
        if n_images_stop is not None and idx >= n_images_stop:
            break
        all_identifiers.append(
            {s: x[s] for s in identifier_strings}
        )

    return all_identifiers


def dictoflists_to_listofdicts(dictoflists):
    n = None
    for k, v in dictoflists.items():
        if n is None:
            n = len(v)
        else:
            assert len(v) == n
    listofdicts = [{k: v[i] for k, v in dictoflists.items()} for i in range(n)]
    return listofdicts


def run_vanilla_evaluation(images, cfg, outputs, image_ids, model=None, exporter=None):
    for img, output, image_id in zip(images, outputs, image_ids):
        if img.shape[2] == 3:
            output_size = img.shape[:2]
        else:
            output_size = img.shape[1:]
        if output['instances'].image_size != output_size:
            # for some reason, it wants an extra dimension...
            B, H, W = output['instances'].pred_masks.shape
            output['instances'].pred_masks = output['instances'].pred_masks.resize(B, 1, H, W)
            # output['instances'] = detector_postprocess(output['instances'], output_size[0], output_size[1])
        # d. Visualize and export Mask R-CNN predictions
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        proposal_score_thresh = None if model is None else model.roi_heads.test_score_thresh
        visualize_single_image_output(img, metadata, pred_instances=output, proposals=None, image_id=str(image_id),
                                      extra_proposal_details=None,
                                      scale=2.0, proposal_score_thresh=proposal_score_thresh, exporter=exporter)


def run_batch_results_visualization(images, cfg, outputs_d, image_ids, model=None, exporter=None,
                                    visualize_just_image=False):
    outputs = outputs_d['outputs']
    proposalss = outputs_d['proposalss']
    extra_proposal_detailss = outputs_d['extra_proposal_details'] if 'extra_proposal_details' in outputs_d else None

    if type(extra_proposal_detailss) is dict:
        extra_proposal_detailss = dictoflists_to_listofdicts(extra_proposal_detailss)

    for img, output, proposals, extra_proposal_details, image_id in \
            zip(images, outputs, proposalss, extra_proposal_detailss, image_ids):
        run_single_image_results_visualization(cfg, exporter, extra_proposal_details, image_id, img, model, output,
                                               proposals, visualize_just_image)


def run_single_image_results_visualization(cfg, exporter, extra_proposal_details, image_id, img, model, output,
                                           proposals, visualize_just_image):
    img, pred_instances, proposals = prep_for_visualization(cfg, img, output['instances'], proposals)
    output['instances'] = pred_instances
    # d. Visualize and export Mask R-CNN predictions
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    proposal_score_thresh = None if model is None else model.roi_heads.test_score_thresh
    visualize_single_image_output(img, metadata, pred_instances=output, proposals=proposals, image_id=str(image_id),
                                  extra_proposal_details=extra_proposal_details,
                                  scale=2.0, proposal_score_thresh=proposal_score_thresh, exporter=exporter,
                                  visualize_just_image=visualize_just_image)


def prep_for_visualization(cfg, img, pred_instances=None, proposals=None):
    if img.shape[2] == 3:
        output_size = img.shape[:2]
    else:
        output_size = img.shape[1:]
    if proposals is not None:
        if proposals.image_size != output_size:
            proposals = detector_postprocess(proposals, output_size[0], output_size[1])
    if pred_instances is not None:
        if pred_instances.image_size != output_size:
            # for some reason, it wants an extra dimension...
            if len(pred_instances.pred_masks.shape) == 3:
                B, H, W = pred_instances.pred_masks.shape
                pred_instances.pred_masks = pred_instances.pred_masks.resize(B, 1, H, W)
            else:
                assert len(pred_instances.pred_masks.shape) == 4
            if len(pred_instances.pred_masks_soft.shape) == 3:
                B, H, W = pred_instances.pred_masks_soft.shape
                pred_instances.pred_masks_soft = pred_instances.pred_masks_soft.resize(B, 1, H, W)
            else:
                assert len(pred_instances.pred_masks_soft.shape) == 4
            pred_instances = detector_postprocess(pred_instances, output_size[0], output_size[1])
    img = input_img_to_rgb(img, cfg)
    return img, pred_instances, proposals


def find_datapoint(dataloader, image_id):
    i = 0
    print(f'Finding image_id={image_id}')
    for ds in dataloader:
        if i % 10 == 0:
            print(i)
        for d in ds:
            if equal_ids(d['image_id'], image_id):
                return d
        i += 1
    raise Exception('{} not found in dataloader'.format(image_id))


def equal_ids(id1, id2):
    return str(id1).rstrip('0') == str(id2).rstrip('0')


def get_datapoint_file(cfg, image_id, cachedir='./output/cache/input/'):
    assert os.path.isdir(cachedir)
    saved_input_file = os.path.join(cachedir, f"input_{image_id}.pt")
    # predictor.model.training = True
    if not os.path.exists(saved_input_file):
        dataloader = build_dataloader(cfg)
        datapoint = find_datapoint(dataloader, image_id)
        torch.save(datapoint, saved_input_file)
        gc.collect()
        del dataloader
    return saved_input_file


def build_dataloader(cfg):
    # dataloaders_eval = {
    #     'val': DefaultTrainer.build_test_loader(cfg,
    #                                             {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s])
    #     for s
    #     in ('train', 'val')
    # }
    train_dataloader = DefaultTrainer.build_train_loader(cfg)
    return train_dataloader


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


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def prep_image(datapoint, cfg):
    image_filename = datapoint['file_name']
    input_image = datapoint['image']
    input_image = np.asarray(input_image.permute(1, 2, 0)[:, :, [2, 1, 0]])
    input_image_from_file = cv2.imread(image_filename)
    input_image = convert_datapoint_to_image_format(input_image, input_image_from_file.shape[:2], cfg)
    return input_image


def run_inference_new(model, inputs):
    previously_training = model.training
    model.eval()
    with inference_context(model), torch.no_grad():
        # Get proposals
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposalss, proposal_lossess = model.proposal_generator(images, features, None)

        # Get instance boxes, masks, and proposal idxs
        outputs, extra_proposal_details = model(inputs, trace_proposals=True)

    if previously_training:
        model.train()

    return {'outputs': outputs,
            'proposalss': proposalss,
            'proposal_lossess': proposal_lossess,
            'extra_proposal_details': extra_proposal_details
            }


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


def visualize_instancewise_predictions(img, instance_outputs, cfg, exporter, tag):
    n_instances = len(instance_outputs)
    for i in range(n_instances):
        instance = instance_outputs[[i]]
        vis_utils.show_prediction(img, {'instances': instance},
                                  metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        exporter.export_gcf(tag + f'_inst{i}')


def activate_head_type(trainer, head_type):
    trainer.model.roi_heads.active_mask_head = head_type
    if head_type == 'custom':
        assert type(trainer.model.roi_heads.mask_heads[trainer.model.roi_heads.active_mask_head]) is \
               CustomMaskRCNNConvUpsampleHeadAPD, 'Not using custom head; head type is {}'.format(type(
            trainer.model.roi_heads.mask_heads[trainer.model.roi_heads.active_mask_head]))
    else:
        assert type(trainer.model.roi_heads.mask_heads[trainer.model.roi_heads.active_mask_head]) \
               is not CustomMaskRCNNConvUpsampleHeadAPD
