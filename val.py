import argparse

import cv2
import os
from multimaskextension.analysis.vis_utils import cv2_imshow, FigExporter
from detectron2.engine import DefaultPredictor

from multimaskextension.train import script_utils
import cv2
import os
import sys
from detectron2.engine import DefaultTrainer
from pprint import pprint
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# DETECTRON2_REPO = '/home/adelgior/code/multi-instance-mask-rcnn-extension/detectron2_repo/'
# if not DETECTRON2_REPO in sys.path:
#     sys.path.append(DETECTRON2_REPO)
# print(sys.path)

from multimaskextension.analysis.vis_utils import cv2_imshow, FigExporter
from multimaskextension.train import script_utils
from detectron2.engine import DefaultPredictor

# Don't know how to avoid importing this
from multimaskextension.model import multi_roi_heads_apd
from multimaskextension.data import registryextension


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def main(trained_logdir, rel_model_pth='checkpoint.pth.tar', config_filepath=None):
    assert os.path.exists(trained_logdir)
    config_filepath = config_filepath or os.path.join(trained_logdir, 'config.yaml')
    assert os.path.exists(config_filepath)

    # I. Load pre-existing Mask R-CNN model
    # a. Get example image to work with
    dataset_dir = "/home/allie/data/datasets/d2s"
    assert os.path.exists(dataset_dir)
    impath = f"{dataset_dir}/images/D2S_99001144.jpg"
    assert os.path.exists(impath), impath
    im = cv2.imread(impath)
    cv2_imshow(im)
    exporter = FigExporter()
    exporter.workspace_dir = '/home/allie/workspace/images'
    exporter.export_gcf('input')

    # b. Load model
    cfg = script_utils.get_custom_maskrcnn_cfg(config_filepath)
    # cfg.MODEL.DEVICE = 'cpu'  # because we're running on espresso root node, no gpu available

    predictor = DefaultPredictor(cfg)

    # c. Get predictions and print boxes
    outputs = predictor(im)

    print(outputs)

    pprint(outputs["instances"].pred_classes)
    pprint(outputs["instances"].pred_boxes)

    # d. Visualize and export predictions
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])
    exporter.export_gcf('prediction')

    # e. Load full dataset

    print('CFG: ')
    print(cfg)
    assert len(cfg.DATASETS.TRAIN) == 1
    dataloaders = {
        s: DefaultTrainer.build_test_loader(cfg, {'train': cfg.DATASETS.TRAIN[0], 'val': cfg.DATASETS.TEST[0]}[s]) for s
        in ('train', 'val')
    }
    print('Number of training images: ', len(dataloaders['train']))
    print('Number of validation images: ', len(dataloaders['val']))

    # # f. Export an example
    # datapoint = None
    # for idx, i in enumerate(train_data_loader):
    #     if idx == 0:
    #         datapoint = i
    #
    # example_img = datapoint['image']
    # data_name = os.path.splitext(os.path.basename(datapoint['file_name']))[0]
    # from detectron2.structures.instances import Instances
    # v = Visualizer(example_img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(v.get_image()[:, :, ::-1])
    # exporter.export_gcf('test image')

    # g. Run Mask R-CNN predictions on all COCO images
    # for each image in training set

    from multimaskextension.train.script_utils import just_inference_on_dataset, get_image_identifiers

    outdir = os.path.join('output', 'test', os.path.basename(trained_logdir.strip(os.path.sep)))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for split, data_loader in dataloaders.items():
        n_points = None  # Set to 10 or so for debugging
        just_inference_on_dataset(predictor.model, data_loader, outdir, n_points)

    # Write file list to path
    filelists = {s: os.path.join(outdir, 'filelist_{}.txt'.format(s))
                 for s in ('train', 'val')}

    for s in ('val', 'train'):
        if not os.path.exists(filelists[s]):  # generate filelist
            identifiers = get_image_identifiers(dataloaders[s], identifier_strings=('file_name', 'image_id'))
            list_of_files = [idnt['file_name'] for idnt in identifiers]
            with open(filelists[s], 'w') as fid:
                fid.write('\n'.join(list_of_files))
                fid.write('\n')
                print('Wrote list of {} files to {}'.format(len(list_of_files), filelists[s]))
        else:
            print('File list already exists at {}'.format(filelists[s]))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained-logdir', required=False, default=None)
    parser.add_argument('--rel-model-pth', required=False, default='checkpoint.pth.tar')
    parser.add_argument('--config-filepath', required=False, default=None,
                        help='Will assume {logdir}/config.yaml')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    # trained_logdir = '/home/allie/afs_directories/espresso/code/multi-instance-mask-rcnn-extension/
    # output/logs/train/train_primary_secondary_full_2020-12-28-225457_VCS-daaf4a9_MAX_ITR
    # -1000000_HEAD_TYPE-custom/'
    main(**args.__dict__)
