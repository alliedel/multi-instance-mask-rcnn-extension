import cv2
import gc
import numpy as np
from PIL import Image

import pickle
from fvcore.common.file_io import PathManager
import copy
from detectron2.config import get_cfg
import subprocess
import os
import torch, torch.distributed, logging, time

from detectron2.engine import DefaultTrainer
from detectron2.evaluation.evaluator import inference_context
from detectron2.data import MetadataCatalog
from detectron2.modeling import detector_postprocess
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

DETECTRON_MODEL_ZOO = os.path.expanduser('~/data/models/detectron_model_zoo')
assert os.path.isdir(DETECTRON_MODEL_ZOO)
DETECTRON_REPO = './detectron2_repo'


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
            existing_state['model'].pop(src_key)
    existing_state['model'].update(extra_state)
    if existing_weights_filename.endswith(".pkl"):
        pickle.dump(existing_state, open(new_weights_filename, 'wb'))
    else:
        torch.save(existing_state, new_weights_filename)


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


def get_custom_maskrcnn_cfg(config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/"
"mask_rcnn_R_50_FPN_3x_APD.yaml"):
    cfg = get_maskrcnn_cfg(config_filepath)
    standard_state_file = cfg.MODEL.WEIGHTS
    ext = os.path.splitext(standard_state_file)[1]
    multiinst_state_file = standard_state_file.replace(f'{ext}', f'_multiinst_heads_apd{ext}')
    make_multihead_state_from_single_head_state(standard_state_file, multiinst_state_file)
    cfg.MODEL.WEIGHTS = multiinst_state_file
    return cfg


def get_maskrcnn_cfg(config_filepath=f"{DETECTRON_REPO}/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    cfg = get_cfg()
    cfg.merge_from_file(config_filepath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the
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
        raise Exception('Predictions outdir {} already exists.  Please delete.')
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
    assert data_loader.batch_sampler.batch_size == 1, NotImplementedError('Only handing case of batch size = 1 for now')
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


def display_instances_on_image(image, instance_output_dict, cfg):
    """
    :param image: numpy array (HxWx3)
    :param instance_output_dict: {
                                  'pred_boxes': (n_instances,4),
                                  'scores': (n_instances,),
                                  'pred_classes': (n_instances,),
                                  'pred_masks': (n_instances, H, W)
                                  }
    :return: Nothing.  Displays on current cv figure.

    """
    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(instance_output_dict["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])


def cv2_imshow(img):
    h = plt.imshow(img)
    return h


class FigExporter(object):
    fig_number = 1
    workspace_dir = '/home/adelgior/workspace/images'
    generated_figures = []

    def export_gcf(self, tag=None, use_number=True):

        if tag is None:
            assert use_number
            basename = '{:06d}.png'.format(self.fig_number)
        else:
            if use_number:
                basename = '{:06d}_{}.png'.format(self.fig_number, tag)
            else:
                basename = '{}.png'.format(tag)

        fname = os.path.join(self.workspace_dir, basename)

        FigExporter.fig_number += 1
        plt.savefig(fname)
        dbprint('Exported {}'.format(fname))
        self.generated_figures.append(fname)


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
        visualize(img, metadata, instances=output, proposals=None, image_id=str(image_id),
                  extra_proposal_details=None,
                  scale=2.0, proposal_score_thresh=proposal_score_thresh, exporter=exporter)


def run_evaluation(images, cfg, outputs_d, image_ids, model=None, exporter=None):
    outputs = outputs_d.pop('outputs')
    proposalss = outputs_d.pop('proposalss')
    extra_proposal_detailss = outputs_d.pop('extra_proposal_details', None)
    if type(extra_proposal_detailss) is dict:
        extra_proposal_detailss = dictoflists_to_listofdicts(extra_proposal_detailss)

    for img, output, proposals, extra_proposal_details, image_id in \
            zip(images, outputs, proposalss, extra_proposal_detailss, image_ids):
        if img.shape[2] == 3:
            output_size = img.shape[:2]
        else:
            output_size = img.shape[1:]
        if proposals.image_size != output_size:
            proposals = detector_postprocess(proposals, output_size[0], output_size[1])
        if output['instances'].image_size != output_size:
            # for some reason, it wants an extra dimension...
            B, H, W = output['instances'].pred_masks.shape
            output['instances'].pred_masks = output['instances'].pred_masks.resize(B, 1, H, W)
            output['instances'] = detector_postprocess(output['instances'], output_size[0], output_size[1])
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        if cfg.INPUT.FORMAT == "BGR":
            img = np.asarray(img[:, :, [2, 1, 0]])
        else:
            img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
        # d. Visualize and export Mask R-CNN predictions
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        proposal_score_thresh = None if model is None else model.roi_heads.test_score_thresh
        visualize(img, metadata, instances=output, proposals=proposals, image_id=str(image_id),
                  extra_proposal_details=extra_proposal_details,
                  scale=2.0, proposal_score_thresh=proposal_score_thresh, exporter=exporter)


def visualize(img, metadata, instances, proposals, image_id, extra_proposal_details=None,
              scale=2.0, map_instance_to_proposal_vis=True, proposal_score_thresh=None, exporter=None):
    if img is not None:
        cv2_imshow(img[:, :, ::-1].astype('uint8'))
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_input')

    if proposals is not None:
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        proposals.pred_boxes = proposals.proposal_boxes
        v = v.draw_instance_predictions(proposals.to('cpu'))
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_proposals')

    if extra_proposal_details is not None:
        selected_proposal_idxs = extra_proposal_details['selected_proposal_idxs']
    else:
        selected_proposal_idxs = None

    if extra_proposal_details is not None:
        extra_proposal_details['scores'] = extra_proposal_details['scores'].to('cpu')
        proposal_subset = (extra_proposal_details['scores'][:, :-1] > proposal_score_thresh).nonzero()
        proposal_subset_inds = proposal_subset[:, 0]
        proposal_subset_classes = proposal_subset[:, 1]
        proposals_past_thresh = proposals[proposal_subset_inds]
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        proposals_past_thresh.pred_boxes = proposals_past_thresh.proposal_boxes
        if map_instance_to_proposal_vis:
            proposals_past_thresh.scores = extra_proposal_details['scores'][
                proposal_subset[:, 0], proposal_subset[:, 1]]
            proposals_past_thresh.pred_classes = proposal_subset_classes

        proposals_past_thresh = proposals_past_thresh.to('cpu')
        v = v.draw_instance_predictions(proposals_past_thresh)
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_proposals_past_thresh')

    if selected_proposal_idxs is not None:
        assert len(selected_proposal_idxs) == len(instances['instances'])
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        proposals_selected = proposals[selected_proposal_idxs]
        proposals_selected.pred_boxes = proposals_selected.proposal_boxes
        if map_instance_to_proposal_vis:
            proposals_selected.scores = instances['instances'].scores
            proposals_selected.pred_classes = instances['instances'].pred_classes

        v = v.draw_instance_predictions(proposals_selected.to('cpu'))
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_selected_proposals')

    if instances is not None:
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        v._default_font_size = v._default_font_size * 1.5
        v = v.draw_instance_predictions(instances["instances"].to("cpu"))
        cv2_imshow(v.get_image()[:, :, ::-1])
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_prediction')


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


def get_datapoint_file(cfg, image_id):
    saved_input_file = f"input_{image_id}.pt"
    # predictor.model.training = True
    if not os.path.exists(saved_input_file):
        dataloader = build_dataloader(cfg)
        datapoint = find_datapoint(dataloader, image_id)
        torch.save(datapoint, saved_input_file)
        gc.collect()
        del dataloader
    return saved_input_file


def collate_figures(figures, figure_name, exporter):
    out_fig = cv2.hconcat([cv2.imread(f) for f in figures])
    fname = os.path.join(exporter.workspace_dir, figure_name + '.png')
    cv2.imwrite(fname.replace('.png', '_fullres.png'), out_fig)
    cv2_imshow(out_fig.astype('uint8'))
    exporter.export_gcf(figure_name)


def build_dataloader(cfg):
    dataloaders_eval = {
        'val': DefaultTrainer.build_test_loader(cfg, {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s])
        for s
        in ('train', 'val')
    }
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
