import argparse
import os
import torch
import json

from tabulate import tabulate

from multimaskextension.train import script_utils
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

from detectron2.engine import DefaultTrainer

# Don't know how to avoid importing this
from multimaskextension.model import multi_roi_heads_apd
from multimaskextension.data import registryextension
from multimaskextension.train.trainer_apd import Trainer_APD
from multimaskextension.analysis.multimaskevaluator import MultiMaskCOCOEvaluator
import pickle


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def build_evaluator(cfg, dataset_name, output_folder=None, distributed=True):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, f"inference_{cfg.DATASETS.TEST}")
    # default inference name
    mask_names = ['pred_masks']
    if cfg.MODEL.ROI_MASK_HEAD.INIT_ACTIVATED_MASK_HEAD != 'standard':
        mask_names.append('pred_masks1')
        mask_names.append('pred_masks2')
    evaluators = [MultiMaskCOCOEvaluator(dataset_name, cfg, distributed, output_folder, mask_names=mask_names)]
    if len(evaluators) == 1:
        return evaluators[0]
    return evaluators


def main(trained_logdir, rel_model_pth='checkpoint.pth.tar', config_filepath=None,
         overwrite_preds=False, cpu=False, val_dataset=None, save_all_predictions=False):
    assert os.path.exists(trained_logdir), trained_logdir
    config_filepath = config_filepath or os.path.join(trained_logdir, 'config.yaml')
    assert os.path.exists(config_filepath), config_filepath

    checkpoint_resume = os.path.join(trained_logdir, rel_model_pth)

    # I. Load pre-existing Mask R-CNN model
    cfg = script_utils.get_custom_maskrcnn_cfg(config_filepath, weights_checkpoint=checkpoint_resume)
    if cpu:
        cfg.MODEL.DEVICE = 'cpu'
    if val_dataset is not None:
        if type(val_dataset) is str:
            val_dataset = [val_dataset]
        cfg.DATASETS.TEST = val_dataset

    model = Trainer_APD.build_model(cfg)

    print('Loading state dict')
    state = torch.load(checkpoint_resume, map_location=torch.device('cpu')) if cpu \
        else torch.load(checkpoint_resume)
    model.load_state_dict(state['model_state_dict'])

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

    outdir = os.path.join('output', 'logs', 'test', os.path.basename(trained_logdir.strip(os.path.sep)),
                          cfg.DATASETS.TEST[0])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    evaluators = build_evaluator(cfg, dataset_name=cfg.DATASETS.TEST[0], output_folder=outdir, distributed=False)
    print('Testing')
    results = Trainer_APD.test(cfg, model, evaluators)
    fname = os.path.join(outdir, 'results.pkl')
    print('Check results at (and its variants):\n{}'.format(fname))
    pickle.dump(results, open(fname, 'wb'))
    json.dump(results, open(fname.replace('.pkl', '.json'), 'w'))
    with open(fname.replace('.pkl', '.txt'), 'w') as f:
        # for dataset_name, results_dic in results.items():
        #     f.write('-- Test dataset: {}'.format(dataset_name))
        for task, res in results.items():
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            f.write('\n')
            f.write("Task: {}".format(task))
            f.write('\n')
            f.write("" + ",".join([k[0] for k in important_res]))
            f.write('\n')
            f.write("" + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))

    if save_all_predictions:
        for split, data_loader in dataloaders.items():
            n_points = None  # Set to 10 or so for debugging
            pred_dir = os.path.join(outdir, f"predictions_{cfg.DATASETS.TEST}")
            if os.path.exists(pred_dir) and not overwrite_preds:
                print(f"{pred_dir} already exists. Skipping inference.")
            else:
                script_utils.just_inference_on_dataset(model, data_loader, outdir, n_points)

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
    parser.add_argument('--trained-logdir', required=True)
    parser.add_argument('--overwrite-preds', required=False, default=None)
    parser.add_argument('--rel-model-pth', required=False, default='checkpoint.pth.tar')
    parser.add_argument('--val-dataset', required=False, default=None, type=lambda arg: arg.split(','),
                        help='Give comma-separated list of validation datasets.  Default: The dataset specified in '
                             'cfg.DATASETS.TEST')
    parser.add_argument('--config-filepath', required=False, default=None,
                        help='Will assume {logdir}/config.yaml')
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--save-all-predictions', default=False, action='store_true')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    # trained_logdir = '/home/allie/afs_directories/espresso/code/multi-instance-mask-rcnn-extension/
    # output/logs/train/train_primary_secondary_full_2020-12-28-225457_VCS-daaf4a9_MAX_ITR
    # -1000000_HEAD_TYPE-custom/'
    main(**args.__dict__)
