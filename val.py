import argparse
import os

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

# DETECTRON2_REPO = '/home/adelgior/code/multi-instance-mask-rcnn-extension/detectron2_repo/'
# if not DETECTRON2_REPO in sys.path:
#     sys.path.append(DETECTRON2_REPO)
# print(sys.path)
<<<<<<< HEAD
from multimaskextension.train.script_utils import just_inference_on_dataset, get_image_identifiers
from multimaskextension.train import script_utils
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor

# Don't know how to avoid importing this
from multimaskextension.model import multi_roi_heads_apd
from multimaskextension.data import registryextension
from multimaskextension.train.trainer_apd import Trainer_APD


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def main(trained_logdir, rel_model_pth='checkpoint.pth.tar', config_filepath=None,
         overwrite_preds=False, cpu=False):
    assert os.path.exists(trained_logdir), trained_logdir
    config_filepath = config_filepath or os.path.join(trained_logdir, 'config.yaml')
    assert os.path.exists(config_filepath), config_filepath

    # I. Load pre-existing Mask R-CNN model
    cfg = script_utils.get_custom_maskrcnn_cfg(config_filepath)
    if cpu:
        cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    checkpoint_resume = os.path.join(trained_logdir, rel_model_pth)
    print('Loading state dict')
    state = torch.load(checkpoint_resume, map_location=torch.device('cpu')) if cpu \
        else torch.load(checkpoint_resume)
    predictor.model.load_state_dict(state['model_state_dict'])

    # e. Load full dataset
    print('CFG: ')
    print(cfg)
    assert len(cfg.DATASETS.TRAIN) == 1
    dataloaders = {
        s: DefaultTrainer.build_test_loader(
            cfg, {
                'train': cfg.DATASETS.TRAIN[0],
                'val': cfg.DATASETS.TEST[0]
            }[s]) for s
        in ('train', 'val')
    }
    print('Number of training images: ', len(dataloaders['train']))
    print('Number of validation images: ', len(dataloaders['val']))

    outdir = os.path.join('output', 'test', os.path.basename(trained_logdir.strip(os.path.sep)))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    evaluators = build_evaluator(cfg, dataset_name=cfg.DATASETS.TEST[0], output_folder=outdir)
    Trainer_APD.test(cfg, predictor.model, evaluators)

    for split, data_loader in dataloaders.items():
        n_points = None  # Set to 10 or so for debugging
        pred_dir = os.path.join(outdir, 'predictions')
        if os.path.exists(pred_dir) and not overwrite_preds:
            print(f"{pred_dir} already exists. Skipping inference.")
        else:
            script_utils.just_inference_on_dataset(predictor.model, data_loader, outdir, n_points)

    # Write file list to path
    filelists = {s: os.path.join(outdir, 'filelist_{}.txt'.format(s))
                 for s in ('train', 'val')}

    for s in ('val', 'train'):
        if not os.path.exists(filelists[s]):  # generate filelist
            identifiers = script_utils.get_image_identifiers(dataloaders[s],
                                                             identifier_strings=('file_name', 'image_id'))
            list_of_files = [idnt['file_name'] for idnt in identifiers]
            with open(filelists[s], 'w') as fid:
                fid.write('\n'.join(list_of_files))
                fid.write('\n')
                print('Wrote list of {} files to {}'.format(len(list_of_files), filelists[s]))
        else:
            print('File list already exists at {}'.format(filelists[s]))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained-logdir', required=True, default=None)
    parser.add_argument('--overwrite-preds', required=False, default=None)
    parser.add_argument('--rel-model-pth', required=False, default='checkpoint.pth.tar')
    parser.add_argument('--config-filepath', required=False, default=None,
                        help='Will assume {logdir}/config.yaml')
    parser.add_argument('--cpu', default=False, action='store_true')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    # trained_logdir = '/home/allie/afs_directories/espresso/code/multi-instance-mask-rcnn-extension/
    # output/logs/train/train_primary_secondary_full_2020-12-28-225457_VCS-daaf4a9_MAX_ITR
    # -1000000_HEAD_TYPE-custom/'
    main(**args.__dict__)
