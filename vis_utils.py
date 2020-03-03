import cv2
import numpy as np
import os
from PIL import Image

from detectron2.layers import paste_masks_in_image

from detectron2.structures import Instances
from matplotlib import pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


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


def visualize_single_image_output(img, metadata, instances, proposals, image_id, extra_proposal_details=None,
                                  scale=2.0, map_instance_to_proposal_vis=True, proposal_score_thresh=None, exporter=None,
                                  visualize_just_image=False):
    assert img is not None
    if visualize_just_image:
        cv2_imshow(img[:, :, ::-1].astype('uint8'))
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_input')

    if proposals is not None:
        show_proposals(img, proposals, metadata, scale)
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_proposals')

    selected_proposal_idxs = None if extra_proposal_details is None else \
        extra_proposal_details['selected_proposal_idxs']

    if extra_proposal_details is not None:
        show_proposals_past_thresh(img, proposals, extra_proposal_details['scores'], proposal_score_thresh, metadata,
                                   map_instance_to_proposal_vis, scale)
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_proposals_past_thresh')

    if selected_proposal_idxs is not None:
        assert len(selected_proposal_idxs) == len(instances['instances'])
        show_selected_proposals(img, instances, proposals, selected_proposal_idxs, map_instance_to_proposal_vis,
                                metadata, scale)
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_selected_proposals')

    if instances is not None:
        show_prediction(img, instances, metadata, scale)
        exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_prediction')


def show_single_instance_prediction(img, metadata, instance, image_id, scale=2.0, exporter=None,
                                    visualize_just_image=False):
    show_prediction(img, {'instances': [instance]}, metadata, scale)


def show_prediction(img, instances, metadata, scale=2.0):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    v._default_font_size = v._default_font_size * 1.5
    v = v.draw_instance_predictions(instances["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])


def show_selected_proposals(img, instances, proposals, selected_proposal_idxs, map_instance_to_proposal_vis, metadata,
                            scale=2.0):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    v._default_font_size = v._default_font_size * 1.5
    proposals_selected = proposals[selected_proposal_idxs]
    proposals_selected.pred_boxes = proposals_selected.proposal_boxes
    if map_instance_to_proposal_vis:
        proposals_selected.scores = instances['instances'].scores
        proposals_selected.pred_classes = instances['instances'].pred_classes
    v = v.draw_instance_predictions(proposals_selected.to('cpu'))
    cv2_imshow(v.get_image()[:, :, ::-1])


def show_proposals_past_thresh(img, proposals, scores, proposal_score_thresh, metadata, map_instance_to_proposal_vis,
                               scale):
    scores = scores.to('cpu')
    proposal_subset = (scores[:, :-1] > proposal_score_thresh).nonzero()
    proposal_subset_inds = proposal_subset[:, 0]
    proposal_subset_classes = proposal_subset[:, 1]
    proposals_past_thresh = proposals[proposal_subset_inds]
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    v._default_font_size = v._default_font_size * 1.5
    proposals_past_thresh.pred_boxes = proposals_past_thresh.proposal_boxes
    if map_instance_to_proposal_vis:
        proposals_past_thresh.scores = scores[proposal_subset[:, 0], proposal_subset[:, 1]]
        proposals_past_thresh.pred_classes = proposal_subset_classes
    proposals_past_thresh = proposals_past_thresh.to('cpu')
    v = v.draw_instance_predictions(proposals_past_thresh)
    cv2_imshow(v.get_image()[:, :, ::-1])


def show_proposals(img, proposals, metadata, scale=2.0, default_size_multiplier=1.5):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    v._default_font_size = v._default_font_size * default_size_multiplier
    proposals.pred_boxes = proposals.proposal_boxes
    v = v.draw_instance_predictions(proposals.to('cpu'))
    cv2_imshow(v.get_image()[:, :, ::-1])


def collate_figures(figures, figure_name, exporter):
    out_fig = cv2.hconcat([cv2.imread(f) for f in figures])
    fname = os.path.join(exporter.workspace_dir, figure_name + '.png')
    fullres_name = fname.replace('.png', '_fullres.png')
    cv2.imwrite(fullres_name, out_fig)
    cv2_imshow(out_fig.astype('uint8'))
    exporter.export_gcf(figure_name)
    return fullres_name


def show_groundtruth(datapoint, cfg, scale=2.0):
    # if predictions.has("pred_masks"):
    #     masks = predictions.pred_masks.numpy()
    #     masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
    # else:
    #     masks = None

    input_image = datapoint['image']
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    visualizer = Visualizer(input_img_to_rgb(input_image, cfg), metadata=metadata, scale=scale)
    names = metadata.get("thing_classes", None)
    labels = [names[i] for i in datapoint['instances'].gt_classes]
    vis = visualizer.overlay_instances(boxes=datapoint['instances'].gt_boxes, labels=labels,
                                       masks=datapoint['instances'].gt_masks,
                                       keypoints=None)
    cv2_imshow(vis.get_image()[:, :, ::-1])


def input_img_to_rgb(img, cfg):
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if cfg.INPUT.FORMAT == "BGR":
        img = np.asarray(img[:, :, [2, 1, 0]])
    else:
        img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
    return img
