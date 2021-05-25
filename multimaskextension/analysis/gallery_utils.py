"""
These could go in visualization_utils.py, but I separated because there were soo many
functions, and this division makes it clearer that these functions are for useful
utils for image-to-image comparisons rather than the more specific purpose of generating the
actual prediction/segmentation images.

The initial reason this file was created was to make an unwieldy ipynb actually readable... (Now
I just import functions from here :) )
"""
import os
from itertools import chain

from PIL import Image, ImageFont, ImageDraw
import numpy as np

from multimaskextension.analysis import imgutils
from multimaskextension.analysis.visualization_utils import centerize, get_max_height_and_width, \
    _tile_images


def blank_image(shp, color=(0, 0, 0)):
    sz = shp[::-1]
    return np.array(Image.new("RGB", sz, color))


def form_a_row(list_of_imgs):
    return np.concatenate(list_of_imgs, axis=1)


def form_a_col(list_of_imgs):
    return np.concatenate(list_of_imgs, axis=0)


def get_tile_image(im_matrix, result_img=None, margin_color=None, margin_size=2):
    """Concatenate images whose sizes are different.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """
    if margin_color is None:
        margin_size = 0
    # get max tile size to which each image should be resized
    max_height, max_width = get_max_height_and_width(chain.from_iterable(im_matrix))
    # resize and concatenate images
    tile_shape = (len(im_matrix), len(im_matrix[0]))
    imgs = []
    for r in range(len(im_matrix)):
        for i, img in enumerate(im_matrix[r]):
            img = imgutils.resize_np_img(img, (max_height, max_width))
            if len(img.shape) == 3:
                img = centerize(img, (max_height + margin_size * 2, max_width + margin_size * 2, 3),
                                margin_color)
            else:
                img = centerize(img, (max_height + margin_size * 2, max_width + margin_size * 2),
                                margin_color)
            imgs.append(img)
    return _tile_images(imgs, tile_shape, result_img)



def concatenate_img(im_matrix, margins=True):
    """
    im_matrix is a list[R][C] of images.  len(im_matrix[r]) must equal C for all r.  Image sizes
    must be compatible with the matrix form of concatenation.
    """
    for r in range(len(im_matrix)):
        row = im_matrix[r]
        rs = [row[c].shape[0] for c in range(len(row))]
        assert all(r_ == rs[0] for r_ in rs), f"im_matrix images dont align by col size: {rs}"

    for c in range(len(im_matrix[0])):
        col = [imr[c] for imr in im_matrix]
        cs = [col[c].shape[1] for c in range(len(col))]
        assert all(c_ == cs[0] for c_ in cs), f"im_matrix images dont align by row size: {cs}"

    ret_img = form_a_col([form_a_row(im_lst) for im_lst in im_matrix])

    return ret_img


def assemble_img_matrix_with_titles(im_matrix, row_titles, col_titles, row_ttl_size=None,
                                    col_ttl_size=None):
    R, C = len(im_matrix), len(im_matrix[0])
    assert all(C == len(im_lst) for im_lst in
               im_matrix), 'Must be a full matrix (leave empty or None if no image belongs there)'
    im_sz = None
    for im in chain.from_iterable(im_matrix):
        if im is None or im == []:
            continue
        if im_sz is None:
            im_sz = im.shape
        else:
            assert all(x == y for x, y in zip(im.shape, im_sz))

    # Fill in empty images
    for row_i, im_lst in enumerate(im_matrix):
        for col_j, im in enumerate(im_lst):
            if im is None or im == []:
                im_matrix[row_i][col_j] = blank_image(im_sz[:2])

    im_sz_for_text = im_sz
    if col_ttl_size is None:
        col_ttl_size = (im_sz_for_text[0] // 2, im_sz_for_text[1])
    if row_ttl_size is None:
        row_ttl_size = (im_sz_for_text[0], im_sz_for_text[1])

    # First row is column titles
    first_row_lst = []
    if col_titles is not None:
        assert len(col_titles) == C
        if row_titles is not None:
            first_row_lst.append(blank_image((col_ttl_size[0], row_ttl_size[1])))
        for col_ttl in col_titles:
            first_row_lst.append(np.array(get_text_img(col_ttl_size, col_ttl)))

    # First column is row titles
    first_col_lst = []
    if row_titles is not None:
        assert len(row_titles) == R
        for row_ttl in row_titles:
            first_col_lst.append(np.array(get_text_img(row_ttl_size, row_ttl)))

    # Prepend each row with its title if needed
    if first_col_lst == []:
        im_matrix_with_titles = im_matrix
    else:
        im_matrix_with_titles = [[first_col_lst[ri]] + im_matrix[ri] for ri in
                                 range(len(im_matrix))]

    # Stack first row if needed
    if first_row_lst != []:
        im_matrix_with_titles = [first_row_lst] + im_matrix_with_titles
    return im_matrix_with_titles


def center_start_loc(im_sz, rect_sz):
    margin_tot = (im_sz[0] - rect_sz[0], im_sz[1] - rect_sz[1])
    return (margin_tot[0] // 2, margin_tot[1] // 2)


def get_font_path():
    pths = ["FreeMono.ttf", "/usr/share/fonts/truetype/freefont/FreeMono.ttf"]
    for pth in pths:
        if os.path.exists(pth):
            return pth
    return None


def get_font(sz, fontpth=None):
    if fontpth is None:
        fontpth = get_font_path()
    return ImageFont.truetype(fontpth, sz, encoding="unic")


def get_font_fontsize_and_loc(im_sz, text, img_fraction=0.80):
    def op_on_size_til_fit(cur, op):
        while (get_font(cur).getsize(text)[0] > img_fraction * im_sz[0]) or (
                get_font(cur).getsize(text)[1] > img_fraction * im_sz[1]):
            cur = op(cur)
            if cur < 1:
                return 1
        return cur

    cur_fontsize = 10000
    cur_fontsize = op_on_size_til_fit(cur_fontsize, lambda x: int(x / 10))
    cur_fontsize = op_on_size_til_fit(cur_fontsize * 10, lambda x: int(x / 2))
    cur_fontsize = op_on_size_til_fit(cur_fontsize * 2, lambda x: x - 1)
    font = get_font(cur_fontsize)
    return font, cur_fontsize, center_start_loc(im_sz, font.getsize(text))


def get_text_img(shp, title_text, textcolor=(255, 255, 255), bgcolor=(0, 0, 0), rot=0):
    sz = shp[::-1]  # PIL format
    img = Image.new("RGB", sz, bgcolor)
    font, fontsize, txt_loc = get_font_fontsize_and_loc(sz, title_text)
    image_editable = ImageDraw.Draw(img)
    image_editable.text(txt_loc, title_text, fill=textcolor, font=font)
    if rot is not 0:
        img = img.rotate(rot, expand=0)
    return img


def zeropad_arrlist(arrlist):
    ndim = len(arrlist[0].shape)
    max_shp = [0 for _ in range(ndim)]
    for arr in arrlist:
        max_shp = [max(arr.shape[i], max_shp[i]) for i in range(ndim)]
    ret_arrs = [np.zeros(max_shp, dtype=arrlist[0].dtype) for _ in arrlist]
    for arrin, arrout in zip(arrlist, ret_arrs):
        arrout[:arrin.shape[0], :arrin.shape[1]] = arrin
    return ret_arrs


def get_mask_name_from_coco_instances_file_name(fname):
    return fname.replace('coco_instances_results_', '')


def get_mask_row_loc(mask_name):
    if mask_name.rstrip('s') == 'pred_mask':
        return 0
    if 'agg' in mask_name:
        return 1
    if mask_name == 'pred_masks1' or mask_name == 'pred_mask1':
        return 2
    if mask_name == 'pred_masks2' or mask_name == 'pred_mask2':
        return 3
    else:
        return 4
