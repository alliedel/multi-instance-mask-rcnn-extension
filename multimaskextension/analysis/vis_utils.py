import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


class FigExporter(object):
    fig_number = 1

    def __init__(self, workspace_dir='/home/adelgior/workspace/images'):
        self.workspace_dir = workspace_dir
        self.generated_figures = []
        self.ext = '.png'
        self.num_fmt = '{:06d}'
        self.delete_individuals_after_collate = False
        self.write_names_on_image_after_collate = True
        self.font_color = (0, 0, 0)
        self.collated_figures = []
        self.collated_figures_fullres = []
        self.previously_generated_figures = []

    @property
    def curr_fig_number_as_str(self):
        return self.num_fmt.format(self.fig_number)

    def export_gcf(self, tag=None, use_number=True):

        fname = self.get_full_impath(tag, use_number)

        FigExporter.fig_number += 1
        plt.savefig(fname)
        dbprint('Exported {}'.format(fname))
        self.generated_figures.append(fname)

    def cv2_imwrite(self, im_arr, tag=None, use_number=True, rgb_to_bgr=True):
        if rgb_to_bgr:
            im_arr = im_arr[:,:,::-1]
        fname = self.get_full_impath(tag, use_number)
        if os.path.splitext(fname)[1] == '':
            fname += '.png'
        FigExporter.fig_number += 1
        cv2.imwrite(fname, im_arr)
        dbprint('Exported {}'.format(fname))
        self.generated_figures.append(fname)

    def get_full_impath(self, tag, use_number):
        if tag is None:
            assert use_number
            basename = self.curr_fig_number_as_str + self.ext
        else:
            if use_number:
                basename = self.curr_fig_number_as_str + '_' + tag + self.ext
            else:
                basename = tag + self.ext
        fname = os.path.join(self.workspace_dir, basename)
        return fname

    def collate_previous(self, out_name, delete_individuals=None):
        delete_individuals = delete_individuals if delete_individuals is not None \
            else self.delete_individuals_after_collate
        fullres_out = collate_figures(self.generated_figures, out_name, self,
                                      delete_individuals=delete_individuals,
                                      write_names_on_image=self.write_names_on_image_after_collate,
                                      font_color=self.font_color)
        # Keep track of what we've generated
        if not self.delete_individuals_after_collate:
            self.previously_generated_figures.extend(self.generated_figures)
        self.generated_figures = []
        self.collated_figures_fullres.append(fullres_out)
        self.collated_figures.append(out_name)
        return fullres_out

    def reset_collate_list(self):
        self.previously_generated_figures.extend(self.generated_figures)
        self.generated_figures = []


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
    plt_imshow(v.get_image())
    return v.get_image()


def plt_imshow(img):
    h = plt.imshow(img)
    return h


def visualize_single_image_output(img, metadata, pred_instances, proposals, image_id,
                                  extra_proposal_details=None,
                                  scale=2.0, map_instance_to_proposal_vis=True,
                                  proposal_score_thresh=None,
                                  exporter: FigExporter = None, visualize_just_image=False,
                                  basename='',
                                  highres=True):
    assert img is not None
    if visualize_just_image:
        if highres:
            exporter.cv2_imwrite(img.astype('uint8'), basename + image_id + '_input')
        else:
            plt_imshow(img)
            exporter.export_gcf(basename + image_id + '_input')

    if proposals is not None:
        imout = show_proposals(img, proposals, metadata, scale)
        if highres:
            exporter.cv2_imwrite(imout.astype('uint8'), basename + image_id + '_proposals')
        else:
            exporter.export_gcf(basename + image_id + '_proposals')

    selected_proposal_idxs = None if extra_proposal_details is None else \
        extra_proposal_details['selected_proposal_idxs']

    if extra_proposal_details is not None:
        imout = show_proposals_past_thresh(img, proposals, extra_proposal_details['scores'],
                                           proposal_score_thresh, metadata,
                                           map_instance_to_proposal_vis, scale)
        if highres:
            exporter.cv2_imwrite(imout.astype('uint8'),
                                 basename + image_id + '_proposals_past_thresh')
        else:
            exporter.export_gcf(basename + image_id + '_proposals_past_thresh')

    if selected_proposal_idxs is not None:
        assert len(selected_proposal_idxs) == len(pred_instances['instances'])
        imout = show_selected_proposals(img, pred_instances, proposals, selected_proposal_idxs,
                                     map_instance_to_proposal_vis, metadata, scale)
        if highres:
            exporter.cv2_imwrite(imout, basename + image_id + '_selected_proposals')
        else:
            exporter.export_gcf(basename + image_id + '_selected_proposals')

    if pred_instances is not None:
        imout = show_prediction(img, pred_instances, metadata, scale)
        if highres:
            exporter.cv2_imwrite(imout,
                os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_prediction')
        else:
            exporter.export_gcf(
                os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_prediction')


def show_prediction(img, instances, metadata, scale=2.0):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    v._default_font_size = v._default_font_size * 1.5
    v = v.draw_instance_predictions(instances["instances"].to("cpu"))
    plt_imshow(v.get_image())
    return v.get_image()


def show_selected_proposals(img, instances, proposals, selected_proposal_idxs,
                            map_instance_to_proposal_vis, metadata, scale=2.0):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    v._default_font_size = v._default_font_size * 1.5
    proposals_selected = proposals[selected_proposal_idxs]
    proposals_selected.pred_boxes = proposals_selected.proposal_boxes
    if map_instance_to_proposal_vis:
        proposals_selected.scores = instances['instances'].scores
        proposals_selected.pred_classes = instances['instances'].pred_classes
    v = v.draw_instance_predictions(proposals_selected.to('cpu'))
    plt_imshow(v.get_image())
    return v.get_image()


def show_proposals_past_thresh(img, proposals, scores, proposal_score_thresh, metadata,
                               map_instance_to_proposal_vis,
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
    plt_imshow(v.get_image())
    return v.get_image()


def show_proposals(img, proposals, metadata, scale=2.0, default_size_multiplier=1.5):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    v._default_font_size = v._default_font_size * default_size_multiplier
    proposals.pred_boxes = proposals.proposal_boxes
    v = v.draw_instance_predictions(proposals.to('cpu'))
    plt_imshow(v.get_image())
    return v.get_image()


def text_fits(text, height, width, font_scale=1.0, font=cv2.FONT_HERSHEY_SIMPLEX, line_type=2):
    text_width, text_height = cv2.getTextSize(text, font, font_scale, line_type)[0]
    return text_width <= width and text_height <= height


def get_font_scale(text, height_lim, width_lim, start_scale=1.0, decr=0.1,
                   font=cv2.FONT_HERSHEY_SIMPLEX, line_type=2):
    assert 0 <= decr < 1.0
    font_scale = start_scale

    while not text_fits(text, height_lim, width_lim, font_scale=font_scale, font=font):
        while not text_fits(text, height_lim, width_lim, font_scale=font_scale, font=font):
            font_scale -= decr
        if font_scale < decr:
            text_width, text_height = cv2.getTextSize(text, font, font_scale, line_type)[0]
            if text_width < 10 or text_height < 10:
                print(Warning(
                    f'Could not find scale to fit text \'{text}\' within ({height_lim}, '
                    f'{width_lim})'))
                return min(start_scale, font_scale / decr)
            decr *= decr

    # dbprint(text, text_height, text_width)
    return font_scale


def get_offset_for_centered_text(text, box_shape_rc, font_scale, font=cv2.FONT_HERSHEY_SIMPLEX,
                                 line_type=2):
    text_width, text_height = cv2.getTextSize(text, font, font_scale, line_type)[0]
    offset_height = (box_shape_rc[0] - text_height) // 2
    offset_width = (box_shape_rc[1] - text_width) // 2
    return offset_height, offset_width


def write_text(img, text, lower_left_rc: (int, int), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
               font_color=(255, 255, 255), line_type=2, enforce_fit_on_image=True,
               enforced_shape_rc=None,
               center_text=None):
    center_text = center_text if center_text is not None else center_text is False
    if enforce_fit_on_image:
        if enforced_shape_rc is None:
            enforced_shape_rc = img.shape
        font_scale = get_font_scale(text, enforced_shape_rc[0], enforced_shape_rc[1],
                                    start_scale=font_scale, font=font)
        if center_text:
            offset_height, offset_width = get_offset_for_centered_text(text, enforced_shape_rc,
                                                                       font_scale)
            lower_left_rc = (lower_left_rc[0] - offset_height, lower_left_rc[1] + offset_width)
    else:
        assert center_text is False, 'Cannot center text if don\'t enforce placement in box'
    cv2.putText(img, text, (lower_left_rc[1], lower_left_rc[0]), font, font_scale, font_color,
                line_type)
    return font_scale


def collate_figures(figures, figure_name, exporter, delete_individuals=False,
                    write_names_on_image=True,
                    font_color=(0, 0, 0), **textkwargs):
    imgs = [cv2.imread(f) for f in figures]
    img_shapes = [i.shape for i in imgs]
    out_fig = cv2.hconcat(imgs)
    del imgs

    if write_names_on_image:
        # titles by default on top 1/10th of image.  We assume for now that all figures are the
        # same size
        start_c = 0
        n_figs = len(figures)
        for imshape, f in zip(img_shapes, figures):
            text = os.path.splitext(os.path.basename(f))[0]
            txt_height = int(imshape[0] / 8)
            txt_width = int(out_fig.shape[1] / n_figs)
            font_scale = write_text(out_fig, text, lower_left_rc=(txt_height, start_c),
                                    enforced_shape_rc=(txt_height, txt_width),
                                    font_color=font_color, center_text=True,
                                    **textkwargs)
            # dbprint(font_scale)
            start_c += imshape[1]
    if len(os.path.splitext(figure_name)[1]) == 0:
        figure_name += '.png'
    fname = os.path.join(exporter.workspace_dir, figure_name)
    fullres_name = fname.replace('.png', '_fullres.png')

    cv2.imwrite(fullres_name, out_fig)
    plt_imshow(out_fig.astype('uint8'))
    exporter.export_gcf(figure_name)

    if delete_individuals:
        assert os.path.exists(fullres_name)
        for figure in figures:
            os.remove(figure)
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
    plt_imshow(vis.get_image())


def input_img_to_rgb(img, cfg):
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if cfg.INPUT.FORMAT == "BGR":
        img = np.asarray(img[:, :, [2, 1, 0]])
    else:
        img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
    return img


def visualize_groundtruth(cfg, datapoint, exporter, image_id):
    show_groundtruth(datapoint, cfg)
    exporter.export_gcf(
        os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_groundtruth')


def visualize_instancewise_groundtruth(datapoint, cfg, exporter, tag=None):
    input_image = input_img_to_rgb(datapoint['image'], cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    names = metadata.get("thing_classes", None)

    boxes = datapoint['instances'].gt_boxes
    masks = datapoint['instances'].gt_masks
    labels = [names[i] for i in datapoint['instances'].gt_classes]

    vis = Visualizer(input_image, metadata=metadata)
    colors = [
        vis._jitter([x / 255 for x in metadata.thing_colors[c]]) for c in
        datapoint['instances'].gt_classes
    ]
    alpha = 0.8
    assert len(labels) == len(masks) == len(boxes)
    for i in range(len(labels)):
        v = Visualizer(input_image, metadata=metadata)
        v = v.overlay_instances(
            masks=masks[i:(i + 1)],
            boxes=boxes[i:(i + 1)],
            labels=labels[i:(i + 1)],
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        plt_imshow(v.get_image())
        figtag = tag + ('_' if tag is not None else '') + f'inst{i}'
        exporter.export_gcf(figtag)
        exporter.cv2_imwrite(v.get_image(), figtag)


def visualize_instancewise_soft_predictions(instance_outputs, exporter: FigExporter, tag):
    n_instances = len(instance_outputs)
    h = plt.figure()
    for i in range(n_instances):
        instance = instance_outputs[[i]]
        soft_mask = instance.pred_masks_soft.cpu().reshape(28, 28)
        assert soft_mask.max() <= 1.0
        assert soft_mask.min() >= 0.0
        plt.gca().imshow(soft_mask, cmap='gray', vmin=0.0, vmax=1.0)
        exporter.export_gcf(tag + f'_inst{i}_soft')
        h.clf()
    plt.close(h)
