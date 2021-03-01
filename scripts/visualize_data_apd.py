## For predictions:
# visualize_data_apd.py
#   --source prediction
#   --predictions-json output/logs/test/train_primary_secondary_full_2021-01-23-185542_VCS-64d87d7_MATCH-0/
#       d2s_val_occlusion/itr256000/coco_instances_results_agg-pred_masks1_pred_masks2.json
#   --dataset d2s_val_occlusion
#   --config-file output/logs/train/train_primary_secondary_full_2021-01-23-185542_VCS-64d87d7_MATCH-1/config.yaml
#   --output-dir /home/adelgior/workspace/images/visualize_data

## For GT:
# visualize_data_apd.py
#   --dataset d2s_val_occlusion
#   --config-file output/logs/train/train_primary_secondary_full_2021-01-23-185542_VCS-64d87d7_MATCH-1/config.yaml
#   --source annotation
#   --output-dir /home/adelgior/workspace/images/visualize_data

import argparse
import json
import os
from itertools import chain

import cv2
import numpy as np
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from multimaskextension.analysis.myvisualizer import MyVisualizer
from multimaskextension.train import script_utils


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader", "prediction"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--dataset", default="train", metavar="DATASET_NAME",
                        help="dataset name from registry, or set to 'train' or 'val' to defaults "
                             "to cfg.DATASETS.TRAIN")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--predictions-json", default="", metavar="json file representing predictions in COCO format",
                        help="path to predictions file if source is set to prediction")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args(in_args)
    if args.source != "prediction":
        assert args.predictions_json == ""
    else:
        assert args.predictions_json != "", '--predictions_file must be set if using --source prediction'
    return args


def json_to_dicts(predictions, iminfo, assumed_bbox_mode=BoxMode.XYXY_ABS):
    """
    input:
        predictions: loaded from json; is in COCO format: one entry per instance
    output:
        predictions_as_dicts: list of per-image dicts of instances in the form
        {'file_name': 'data/datasets/d2s/images/D2S_025300.jpg',
         'height': 1440,
         'width': 1920,
         'image_id': 25300,
         'annotations': [{'iscrowd': 0,
           'bbox': [702.0, 0.0, 721.0, 755.0],
           'category_id': 40,
           'segmentation': {'counts': 'PVkn05k\\14L1O1O1je0',
            'size': [1440, 1920]},
           'bbox_mode': <BoxMode.XYWH_ABS: 1>},

    """
    predictions_as_dicts = {
        anno['image_id']:
            {
                'file_name': anno['file_name'],
                'height': anno['height'],
                'width': anno['width'],
                'image_id': anno['image_id'],
                'annotations': []
            }
        for anno in iminfo
    }
    # we're going to fill it as a dictionary; will convert to list.
    for p in predictions:
        image_id = p['image_id']
        assert image_id in predictions_as_dicts, ValueError(f"predictions didn\'t match the file info from the "
                                                            f"dataset ({image_id} in predictions)")
        if 'bbox_mode' not in p:
            p['bbox_mode'] = assumed_bbox_mode
        predictions_as_dicts[image_id]['annotations'].append(p)
    return list(predictions_as_dicts.values())


def main():
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = script_utils.get_custom_maskrcnn_cfg(args.config_file)
    cfg.DATASETS.TRAIN = {'train': cfg.DATASETS.TRAIN, 'val': cfg.DATASETS.TEST}[args.dataset] \
        if args.dataset in ['train', 'val'] else [args.dataset]
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 2.0 if args.show else 1.0
    if args.source == "dataloader":
        train_data_loader = build_detection_train_loader(cfg)
        for batch in train_data_loader:
            for per_image in batch:
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
                output(vis, str(per_image["image_id"]) + ".jpg")
    elif args.source == "annotation":
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        for dic in dicts:
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = MyVisualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))
    elif args.source == "prediction":
        gt_dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        preds_json = json.load(open(args.predictions_json, 'r'))
        dicts = json_to_dicts(preds_json, iminfo=gt_dicts)
        for dic in dicts:
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = MyVisualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))
    else:
        raise ValueError("Debug error with --source arg")


if __name__ == "__main__":
    main()
