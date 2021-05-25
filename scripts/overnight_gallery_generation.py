## Running overnight on kalman to generate gallery images

## Imports

# **** Use activate_virtualenv detectron2!!!


import os
import sys
from PIL import Image, ImageFont, ImageDraw
from matplotlib.pyplot import imshow
import numpy as np
from matplotlib import pyplot as plt
import tqdm

for source_dir in ['/home/adelgior/afs_directories/kalman/repos/multi-instance-mask-rcnn'
                   '-extension/',
                   '/home/allie/repos/multi-instance-mask-rcnn-extension/detectron2_repo/']:
    if source_dir not in sys.path and os.path.exists(source_dir):
        sys.path.append(source_dir)

# Handling local vs kalman
PROJECT_ROOT = None
for project_root in ['/home/adelgior/code/multi-instance-mask-rcnn-extension/',
                     '/home/adelgior/afs_directories/kalman/repos/multi-instance-mask-rcnn'
                     '-extension']:
    if os.path.exists(project_root):
        PROJECT_ROOT = project_root
assert PROJECT_ROOT is not None, 'Please assign the project root here'
os.environ['DATAPATH'] = os.path.join(project_root, 'data/')

import argparse
import json
# import os
from itertools import chain

import cv2
import numpy as np
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

# %cd /home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/

from multimaskextension.train import script_utils
from multimaskextension.analysis.myvisualizer import MyVisualizer
from multimaskextension.analysis import gallery_utils

# predictions_json = '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension
# /output/logs/test/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1
# /coco_2017_val/itr36000/coco_instances_results_pred_masks1.json'
# predictions_json = '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension
# /output/logs/test/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1
# /coco_2017_val/itr36000/coco_instances_results_agg-pred_masks1_pred_masks2.json'
# predictions_json = '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension
# /output/logs/test/train_2021-03-29-232845_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-None
# /coco_2017_val/itr64000/coco_instances_results_agg-pred_masks.json'
# predictions_json = '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension
# /output/logs/test/train_2021-03-29-232845_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-None
# /coco_2017_val/itr0/coco_instances_results_pred_masks.json'
predictions_json = 'output/logs/test/dummytest/coco_2017_val/itr0' \
                   '/coco_instances_results_pred_masks.json'
model_testdir = os.path.dirname(predictions_json)


# predictions_json = 'coco_instances_results_pred_masks.json'
# predictions_json = 'coco_instances_results_pred_masks.json'
class sample_args:
    config_file = os.path.join(model_testdir, 'config_resume.yaml')
    config_files_lst = None  # gets filled in later; simple list: one per model
    dataset = 'val'
    show = False
    source = 'prediction'
    opts = []
    output_dir = '/home/allie/workspace/images/sample_model_comparison_gallery'
    predictions_json = predictions_json


class args:
    dataset = 'val'
    show = False
    source = 'prediction'
    opts = []
    output_dir = '/home/allie/workspace/images/2021_05_05_model_comparison_gallery'
    predictions_json_lstlst = None  # to be filled in later; nested list: one list per model of
    # length Mi, where Mi is number of masks evaluated for model i
    image_id_ordered_subset = None


def output(vis, fname, caption=None):
    if args.show:
        print(fname)
        if caption:
            print(caption)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()
    #         cv2.imshow("window", )
    #         cv2.waitKey()
    else:
        filepath = os.path.join(args.output_dir, fname)
        if caption:
            caption_fpath = filepath.replace(os.path.splitext(filepath)[1], '.txt')
            if os.path.isfile(caption_fpath):
                os.remove(caption_fpath)
                with open(caption_fpath, 'w') as f:
                    f.write(caption)
        # print(f"Saving to {filepath}")
        vis.save(filepath)


cfg = script_utils.get_custom_maskrcnn_cfg(sample_args.config_file)
# Single prediction: just testing output on a few images

# %matplotlib inline
from matplotlib import pyplot as plt

n_images = 5

cfg.DATASETS.TRAIN = {'train': cfg.DATASETS.TRAIN, 'val': cfg.DATASETS.TEST}[args.dataset] \
    if args.dataset in ['train', 'val'] else [args.dataset]
cfg.merge_from_list(args.opts)
os.makedirs(args.output_dir, exist_ok=True)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


scale = 2.0 if args.show else 1.0
if args.source == "dataloader":
    train_data_loader = build_detection_train_loader(cfg)
    for batch in train_data_loader:
        if n_images < 0:
            break
        for per_image in batch:
            if n_images < 0:
                break
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0)
            if cfg.INPUT.FORMAT == "BGR":
                img = img[:, :, [2, 1, 0]]
            else:
                img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

            visualizer = MyVisualizer(img, metadata=metadata, scale=scale)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            print(per_image['image_id'])
            output(vis, str(per_image["image_id"]) + ".jpg")
            n_images -= 1
elif args.source == "annotation":
    dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
    for dic in dicts:
        img = utils.read_image(dic["file_name"], "RGB")
        visualizer = MyVisualizer(img, metadata=metadata, scale=scale)
        vis = visualizer.draw_dict(dic)
        output(vis, os.path.basename(dic["file_name"]))
elif args.source == "prediction":
    gt_dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
    preds_json = json.load(open(sample_args.predictions_json, 'r'))
    dicts = MyVisualizer.json_to_dicts(preds_json, iminfo=gt_dicts)
    for dic in dicts:
        if n_images < 0:
            break
        img = utils.read_image(dic["file_name"], "RGB")
        visualizer = MyVisualizer(img, metadata=metadata, scale=scale)
        vis = visualizer.draw_dict(dic)
        thing_counts = {}
        for d in dic['annotations']:
            cname = metadata.thing_classes[
                metadata.thing_dataset_id_to_contiguous_id[d['category_id']]]
            if cname not in thing_counts:
                thing_counts[cname] = 1
            else:
                thing_counts[cname] += 1
        print(dic['file_name'], str(thing_counts))
        output(vis, os.path.basename(dic["file_name"]),
               caption=os.path.basename(dic["file_name"]) + ' ' + str(thing_counts))
        n_images -= 1
else:
    raise ValueError("Debug error with --source arg")
print('Done')

import glob

# model_test_dirpths = sorted(glob.glob(
#     'output/logs/test/*'))
model_test_dirpths = [
    #     '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/output/logs
    #     /test/train_2021-03-18-085316_VCS-e05e8a1_MAX_ITR-270000_HEAD_TYPE-custom_MATCH-0',
    'output/logs/test'
    '/train_2021-03-27-232834_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-0',
    'output/logs/test'
    '/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1',
    'output/logs/test/dummytest',
    #     '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/output/logs
    #     /test/train_2021-03-29-232845_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-None',
    #     '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/output/logs
    #     /test/train_primary_secondary_full_2021-01-23-185542_VCS-64d87d7_MATCH-0',
    #     '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/output/logs
    #     /test/train_primary_secondary_full_2021-01-23-185542_VCS-64d87d7_MATCH-1',
    #     '/home/allie/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/output/logs
    #     /test/train_primary_secondary_full_2021-01-24-074402_VCS-64d87d7_MAX_ITR
    #     -270000_HEAD_TYPE-standard'
]
channel_eval_pths = []
for model_test_dirpth in model_test_dirpths:
    regx = os.path.join(model_test_dirpth, '*_val', 'itr36000', '*.json')
    pths = sorted(glob.glob(regx))
    if len(pths) == 0:
        print(f"No matches found in {regx}")
    channel_eval_pths.append(pths)

# Nested list based on how they should be grouped
args.predictions_json_lstlst = [
    [
        'output/logs'
        '/test/dummytest/coco_2017_val/itr0/coco_instances_results_agg-pred_masks.json',
        'output/logs'
        '/test/dummytest/coco_2017_val/itr0/coco_instances_results_pred_masks.json',
    ],
    [
        'output/logs'
        '/test/train_2021-03-27-232834_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-0'
        '/coco_2017_val/itr36000/coco_instances_results_agg-pred_masks1_pred_masks2.json',
        'output/logs'
        '/test/train_2021-03-27-232834_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-0'
        '/coco_2017_val/itr36000/coco_instances_results_pred_masks1.json',
        'output/logs'
        '/test/train_2021-03-27-232834_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-0'
        '/coco_2017_val/itr36000/coco_instances_results_pred_masks2.json',
        ],
    [
        'output/logs'
        '/test/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1'
        '/coco_2017_val/itr36000/coco_instances_results_agg-pred_masks1_pred_masks2.json',
        'output/logs'
        '/test/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1'
        '/coco_2017_val/itr36000/coco_instances_results_pred_masks1.json',
        'output/logs'
        '/test/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1'
        '/coco_2017_val/itr36000/coco_instances_results_pred_masks2.json',
    ],
]

config_files_lstlst = [
    [os.path.abspath(os.path.join(os.path.dirname(predictions_json), 'config_resume.yaml')) for
     predictions_json in lst] for lst in args.predictions_json_lstlst]
config_files_lst = []
for lst in config_files_lstlst:
    assert all(l == lst[0] for l in lst)
    config_files_lst.append(lst[0])

args.config_files_lst = config_files_lst  # Update args

# Setup for multi-mask/model comparison (grid view) -- loading all the dicts for comparison.



def get_thing_counts(dic):
    thing_counts = {}
    for d in dic['annotations']:
        cname = metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[d['category_id']]]
        if cname not in thing_counts:
            thing_counts[cname] = 1
        else:
            thing_counts[cname] += 1


opts = args.opts
dataset = args.dataset
source = args.source
output_dir = args.output_dir
predictions_json_lstlst = args.predictions_json_lstlst
image_ids = None
dicts_lstlst = []
for modeli, config_file in enumerate(config_files_lst):
    print(f"Loading predictions from {modeli}")
    cfg = script_utils.get_custom_maskrcnn_cfg(config_file)
    cfg.DATASETS.TRAIN = {'train': cfg.DATASETS.TRAIN, 'val': cfg.DATASETS.TEST}[dataset] \
        if dataset in ['train', 'val'] else [args.dataset]
    cfg.merge_from_list(opts)
    os.makedirs(output_dir, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    dicts_lst = []
    for maskj, predictions_json in enumerate(predictions_json_lstlst[modeli]):
        if source == "annotation":
            dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        elif source == "prediction":
            gt_dicts = list(
                chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
            preds_json = json.load(open(predictions_json, 'r'))
            dicts = MyVisualizer.json_to_dicts(preds_json, iminfo=gt_dicts)
        else:
            raise Exception
        if image_ids is None:
            image_ids = [dic['image_id'] for dic in dicts]
        else:
            assert all(imid == dic['image_id'] for dic, imid in zip(dicts, image_ids))
        dicts_lst.append(dicts)
    dicts_lstlst.append(dicts_lst)

# Get image_id_ordered_subset
image_id_ordered_subset = args.image_id_ordered_subset
if image_id_ordered_subset is None:
    image_id_ordered_subset = [dic['image_id'] for dic in dicts_lstlst[0][0]][::-1]

initial_image_ids = [dic['image_id'] for dic in dicts_lstlst[0][0]]
sorted_order_idxs = []
for image_id in image_id_ordered_subset:
    idx = initial_image_ids.index(image_id)
    sorted_order_idxs.append(idx)

# Reorder
sorted_dicts_lstlst = [[[dicts[i] for i in sorted_order_idxs] for dicts in dicts_lst] for dicts_lst
                       in dicts_lstlst]

for i, sorted_dicts in enumerate(chain.from_iterable(sorted_dicts_lstlst)):
    assert all(
        imageid == dic['image_id'] for imageid, dic in zip(image_id_ordered_subset, sorted_dicts))
    print('Verified imageids match specified sorted order', i)

# Verify all dic image ids are in the required order

assert len(config_files_lst) == len(sorted_dicts_lstlst)
assert len(sorted_dicts_lstlst[0][0]) == len(sorted_order_idxs), (
len(sorted_dicts_lstlst[0]), len(sorted_order_idxs))

# Multi-model/mask comparison: Different masks are different columns; different models are
# different rows.
# (One model's mask predictions take up one row)

# %matplotlib inline
# shp = vis.get_image().shape[:2]
# col_ttl = 'col title'
# col_ttl_img = gallery_utils.get_text_img((shp[0] // 3, shp[1]), col_ttl)
# plt.imshow(gallery_utils.form_a_col([col_ttl_img, vis.get_image()]))
# row_ttl = 'row title'
# row_ttl_img = gallery_utils.get_text_img((shp[0], shp[1] // 3), row_ttl, rot=90)
# # plt.imshow(np.concatenate([row_ttl_img, vis.get_image()], axis=1))
# plt.imshow(gallery_utils.form_a_row([row_ttl_img, vis.get_image()]))
# Test arguments
# test_im_matrix = [[np.array(vis.get_image()) for i in range(2)] for j in range(3)]
# R, C = len(test_im_matrix), len(test_im_matrix[0])
# test_row_titles = [f"row {i}" for i in range(R)]
# test_col_titles = [f"col {j}" for j in range(C)]
#
# im_matrix = test_im_matrix
# row_titles = test_row_titles
# col_titles = test_col_titles
#
# im_matrix_with_titles = gallery_utils.assemble_img_matrix_with_titles(im_matrix, row_titles,
# col_titles)
#
# mat_as_single_img = gallery_utils.concatenate_img(im_matrix_with_titles)
#
# output(Image.fromarray(mat_as_single_img), "test_img.jpg")
# plt.imshow(mat_as_single_img)
import re

def get_mask_name_from_pred_json_file(pred_json_file):
    mask_name = os.path.splitext(os.path.basename(pred_json_file))[0].replace(
        'coco_instances_results_', '')
    return mask_name


all_mask_names = [get_mask_name_from_pred_json_file(pred_file) for pred_file in
                  chain.from_iterable(args.predictions_json_lstlst)]
unique_names = list(set(all_mask_names))
mask_name_ord = [gallery_utils.get_mask_row_loc(nm) for nm in unique_names]
masks_in_order = [unique_names[i] for i in np.argsort(mask_name_ord)]
col_titles = masks_in_order

print(f"Output dir: {args.output_dir}")
for image_i in tqdm.tqdm(range(len(image_id_ordered_subset)), total=len(image_id_ordered_subset),
                         desc='Generating & saving model comparison images'):
    image_id = image_id_ordered_subset[image_i]
    im_shape = None
    im_matrix = []
    row_titles = []
    for i_cfg, (config_file, predictions_json_files, sorted_dicts_lst) in enumerate(
            zip(args.config_files_lst, args.predictions_json_lstlst, sorted_dicts_lstlst)):
        cfg = script_utils.get_custom_maskrcnn_cfg(config_file)
        itr = os.path.basename(os.path.dirname(config_file))
        model_dir = os.path.basename(
            os.path.abspath(os.path.join(os.path.dirname(config_file), '../../')))
        # print(model_dir)
        if 'dummy' in model_dir:
            cfg_list = ['Mask_RCNN_pretrained']
        else:
            cfg_list = re.split(r'-', model_dir)[6:] + [itr]
        row_name = '-'.join(cfg_list)
        row_titles.append(row_name)
        row = [[] for _ in masks_in_order]
        for pred_json_file, sorted_dicts in zip(predictions_json_files, sorted_dicts_lst):
            mask_name = get_mask_name_from_pred_json_file(pred_json_file)
            col_idx = masks_in_order.index(mask_name)
            dic = sorted_dicts[image_i]
            assert dic['image_id'] == image_id
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = MyVisualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dict(dic)
            im_shape = (img.shape[0], img.shape[1])
            row[col_idx] = np.array(vis.get_image())
        im_matrix.append(row)

    # Fill empties with zeros
    for ri in range(len(im_matrix)):
        for ci, rc in enumerate(im_matrix[ri]):
            if rc == []:
                im_matrix[ri][ci] = gallery_utils.blank_image(im_shape)
    im_matrix_with_titles = gallery_utils.assemble_img_matrix_with_titles(im_matrix, row_titles,
                                                                          col_titles)
    mat_as_single_img = gallery_utils.get_tile_image(im_matrix_with_titles,
                                                     margin_color=(255, 255, 255), margin_size=10)
    output(Image.fromarray(mat_as_single_img), f"{image_id}.jpg")
    # plt.imshow(mat_as_single_img)
