"""
Forked from site-packages/fcn/utils.py
"""
from __future__ import division

import math
import os
import warnings
from os import path as osp

from multimaskextension.analysis import exportutils as export_utils
from . import imgutils


try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import scipy.ndimage
import six
import skimage.color

DEBUG_ASSERTS = True


# -----------------------------------------------------------------------------
# Chainer Util
# -----------------------------------------------------------------------------


def batch_to_vars(batch, device=-1):
    import chainer
    from chainer import cuda
    in_arrays = [np.asarray(x) for x in zip(*batch)]
    if device >= 0:
        in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
    in_vars = [chainer.Variable(x) for x in in_arrays]
    return in_vars


# -----------------------------------------------------------------------------
# Color Util
# -----------------------------------------------------------------------------

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(*args, **kwargs):
    warnings.warn('labelcolormap is renamed to label_colormap.',
                  DeprecationWarning)
    return label_colormap(*args, **kwargs)


def label_colormap(N=256):
    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap




def visualize_labelcolormap(*args, **kwargs):
    warnings.warn(
        'visualize_labelcolormap is renamed to visualize_label_colormap',
        DeprecationWarning)
    return visualize_label_colormap(*args, **kwargs)


def visualize_label_colormap(cmap):
    n_colors = len(cmap)
    ret = np.zeros((n_colors, 10 * 10, 3))
    for i in six.moves.range(n_colors):
        ret[i, ...] = cmap[i]
    return ret.reshape((n_colors * 10, 10, 3))


def get_label_colortable(n_labels, shape):
    if cv2 is None:
        raise RuntimeError('get_label_colortable requires OpenCV (cv2)')
    rows, cols = shape
    if rows * cols < n_labels:
        raise ValueError
    cmap = label_colormap(n_labels)
    table = np.zeros((rows * cols, 50, 50, 3), dtype=np.uint8)
    for lbl_id, color in enumerate(cmap):
        color_uint8 = (color * 255).astype(np.uint8)
        table[lbl_id, :, :] = color_uint8
        text = '{:<2}'.format(lbl_id)
        cv2.putText(table[lbl_id], text, (5, 35),
                    cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    table = table.reshape(rows, cols, 50, 50, 3)
    table = table.transpose(0, 2, 1, 3, 4)
    table = table.reshape(rows * 50, cols * 50, 3)
    return table


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def pad_image_to_right_and_bottom(src, dst_height=None, dst_width=None):
    if dst_width is None and dst_height is None:
        raise ValueError('Height or width required when padding')
    dst_height = dst_height if dst_height is not None else src.shape[0]
    dst_width = dst_width if dst_width is not None else src.shape[1]
    if len(src.shape) == 3:
        dst_shape = (dst_height, dst_width, src.shape[2])
    else:
        assert len(src.shape) == 2
        dst_shape = (dst_height, dst_width)

    # if src.shape == dst_shape:
    #     return src
    padded = np.zeros(dst_shape, dtype=src.dtype)
    padded[:src.shape[0], :src.shape[1], ...] = src
    return padded


def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size

    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    try:
        if len(src.shape) == 3:
            centerized[pad_vertical:(pad_vertical + h), pad_horizontal:(pad_horizontal + w), :] = src
        else:
            centerized[pad_vertical:(pad_vertical + h), pad_horizontal:(pad_horizontal + w)] = src
    except:
        import ipdb; ipdb.set_trace()
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, 3), dtype=np.uint8)
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in six.moves.range(y_num):
        for x in six.moves.range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y * one_height:(y + 1) * one_height,
                x * one_width:(x + 1) * one_width] = imgs[i]
    return concatenated_image


def _tile_same_height_images_into_row(imgs, concatenated_image, concat_direction='horizontal'):
    """Concatenate images whose sizes are same.
    concat_dim: 1 if you want to concatenate along the image x-axis (img1-img2)
                0 if you want to stack vertically
    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    assert concat_direction in ('h', 'v', 'horizontal', 'vertical')
    shared_dim = 0 if concat_direction in ('horizontal', 'h') else 1
    shared_dim_val = imgs[0].shape[shared_dim]
    assert all([i.shape[shared_dim] == shared_dim_val for i in imgs]), \
        'Image shapes, attempting to concat {}ly: {}'.format(concat_direction, [i.shape for i in imgs])

    if concatenated_image is None:
        all_h = shared_dim_val if shared_dim is 0 else sum([i.shape[0] for i in imgs])
        all_w = shared_dim_val if shared_dim is 1 else sum([i.shape[1] for i in imgs])
        shp = (all_h, all_w, 3) if len(imgs[0].shape) == 3 else (all_h, all_w)
        concatenated_image = np.zeros(shp, dtype=np.uint8)

    for idx, img in enumerate(imgs):
        if shared_dim == 0:
            w = img.shape[1]
            concatenated_image[:, idx * w:(idx + 1) * w] = img
        else:
            h = img.shape[0]
            concatenated_image[idx * h:(idx + 1) * h, :] = img
    return concatenated_image


def get_tile_image_1d(imgs, concat_direction='horizonal', result_img=None, margin_color=None, margin_size=2):
    if margin_color is None:
        margin_size = 0
    # resize and concatenate images
    for i, img in enumerate(imgs):
        if len(img.shape) == 3:
            h, w, _ = img.shape
            img = centerize(img, (h + margin_size * 2, w + margin_size * 2, 3),
                            margin_color)
        else:
            h, w = img.shape
            img = centerize(img, (h + margin_size * 2, w + margin_size * 2),
                            margin_color)
        imgs[i] = img
    return _tile_same_height_images_into_row(imgs, result_img, concat_direction=concat_direction)


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None, margin_size=2):
    """Concatenate images whose sizes are different.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return x_num, y_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    if margin_color is None:
        margin_size = 0
    # get max tile size to which each image should be resized
    max_height, max_width = get_max_height_and_width(imgs)
    # resize and concatenate images
    for i, img in enumerate(imgs):
        img = imgutils.resize_np_img(img, (max_height, max_width))
        if len(img.shape) == 3:
            img = centerize(img, (max_height + margin_size * 2, max_width + margin_size * 2, 3),
                            margin_color)
        else:
            img = centerize(img, (max_height + margin_size * 2, max_width + margin_size * 2),
                            margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)


def get_max_height_and_width(imgs):
    max_height, max_width = 0, 0
    for img in imgs:
        max_height = max([max_height, img.shape[0]])
        max_width = max([max_width, img.shape[1]])
    return max_height, max_width


def label2rgb(lbl, img=None, label_names=None, n_labels=None,
              alpha=0.3, thresh_suppress=0):
    if label_names is None:
        if n_labels is None:
            n_labels = lbl.max() + 1  # +1 for bg_label 0
    else:
        if n_labels is None:
            n_labels = len(label_names)
        else:
            assert n_labels == len(label_names), 'n_labels: {}, len(label_names): {}'.format(n_labels, len(label_names))
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = skimage.color.rgb2gray(img)
        img_gray = skimage.color.gray2rgb(img_gray)
        img_gray *= 255
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    if label_names is None:
        return lbl_viz

    # cv2 is required only if label_names is not None
    import cv2
    if cv2 is None:
        warnings.warn('label2rgb with label_names requires OpenCV (cv2), '
                      'so ignoring label_names values.')
        return lbl_viz

    np.random.seed(1234)
    for label in np.unique(lbl):
        if label == -1:
            continue  # unlabeled

        mask = lbl == label
        if 1. * mask.sum() / mask.size < thresh_suppress:
            continue
        mask = (mask * 255).astype(np.uint8)
        y, x = scipy.ndimage.center_of_mass(mask)
        y, x = map(int, [y, x])

        if lbl[y, x] != label:
            Y, X = np.where(mask)
            point_index = np.random.randint(0, len(Y))
            y, x = Y[point_index], X[point_index]

        text = label_names[label]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)

        def get_text_color(color):
            if color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 170:
                return (0, 0, 0)
            return (255, 255, 255)

        color = get_text_color(lbl_viz[y, x])
        cv2.putText(lbl_viz, text,
                    (x - text_size[0] // 2, y),
                    font_face, font_scale, color, thickness)
    return lbl_viz


def my_copy(lst_or_tensor):
    if type(lst_or_tensor) is list:
        return list.copy(lst_or_tensor)
    elif type(lst_or_tensor) is np.ndarray:
        return lst_or_tensor.copy()
    else:
        return lst_or_tensor.clone()


def pad_to_same_size(imgs):
    max_height, max_width = get_max_height_and_width(imgs)
    return [pad_image_to_right_and_bottom(im, max_height, max_width) for im in imgs]


def scores2d2heatmap(scores_single_channel, clims=None, color=(255, 255, 255)):
    M, N = scores_single_channel.shape
    heatmap = np.repeat(scores_single_channel[:, :, np.newaxis], 3, axis=2).astype(np.float32)  # / 255
    if clims is None:
        clims = (heatmap.min(), heatmap.max())
    else:
        clims = (float(clims[0]), float(clims[1]))
    heatmap = (heatmap - clims[0]) / (clims[1] - clims[0])
    heatmap = heatmap * np.array(color).squeeze()[np.newaxis, np.newaxis, :]
    return heatmap


def write_word_in_img_center(img, text, **kwargs):
    r, c = (img.shape[0] // 2), (img.shape[1] // 2)
    write_word_in_location(img, text, r, c, **kwargs)


def write_word_in_location(img, text, r, c, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=None):
    thickness = int(round(thickness or font_scale * 2.0 / 0.7))  # make bolder as it gets larger (scale=0.7 <->
    # thickness=2)
    y, x = r, c
    y, x = map(int, [y, x])
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    color = get_text_color(img[y, x])
    cv2.putText(img, text, (x - text_size[0] // 2, y), font_face, font_scale, color, thickness)


def get_text_color(bg_color):
    """
    Decides whether to write on background in white or black based on background color
    """
    if bg_color[0] * 0.299 + bg_color[1] * 0.587 + bg_color[2] * 0.114 > 170:
        return (0, 0, 0)
    return (255, 255, 255)


def write_image(out_file, out_img):
    try:
        imgutils.write_np_array_as_img(out_img, out_file)
    except ValueError:
        print('size, shape of out_img: {}'.format(type(out_img), out_img.shape))
        raise


def write_label(out_file, out_lbl):
    out_img = label2rgb(out_lbl)
    try:
        imgutils.write_np_array_as_img(out_img, out_file)
    except ValueError:
        print('size, shape of out_img: {}'.format(type(out_img), out_img.shape))
        raise


def export_visualizations(visualizations, out_dir, tensorboard_writer, iteration, basename='val_',
                          tile=True, ext='.png'):
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    if tile:
        out_img = get_tile_image(visualizations, margin_color=[255, 255, 255],
                                 margin_size=50)
        tag = '{}images'.format(basename)
        if tensorboard_writer is not None:
            export_utils.log_images(tensorboard_writer, tag, [out_img], iteration, numbers=[0])
        out_subdir = osp.join(out_dir, tag)
        if not osp.exists(out_subdir):
            os.makedirs(out_subdir)
        out_file = osp.join(out_subdir, 'iter-%012d' % iteration + ext)
        write_image(out_file, out_img)
    else:
        tag = '{}images'.format(basename)
        out_subdir = osp.join(out_dir, tag)
        if not osp.exists(out_subdir):
            os.makedirs(out_subdir)
        for img_idx, out_img in enumerate(visualizations):
            if tensorboard_writer is not None:
                export_utils.log_images(tensorboard_writer, tag, [out_img], iteration,
                                        numbers=[img_idx])
            out_subsubdir = osp.join(out_subdir, str(img_idx))
            if not osp.exists(out_subsubdir):
                os.makedirs(out_subsubdir)
            out_file = osp.join(out_subsubdir, 'iter-%012d' % iteration + ext)
            write_image(out_file, out_img)
