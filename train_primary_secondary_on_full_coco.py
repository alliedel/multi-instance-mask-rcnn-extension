"""
The goal of :train_primary_secondary_on_full_coco: is to run primary-secondary mask training on COCO.
We are changing as little as possible.  The only things we're making extra effort on compared to
previous train so far are:
- Good dataloader (with pinned memory)
- Saving intermediate models
- Running a 'smart' optimizer (borrowed from Mask R-CNN)

Things left for other scripts:
- Good parameter loading
- Logging (text and tensorboard)
"""
import os
import argparse
import local_pyutils
import torch

import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger

from multimaskextension.train import script_utils
from multimaskextension.train.trainer_apd import Trainer_APD

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def main(args):
    config_filepath = args.config_filepath  #
    # './detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_APD.yaml',
    resume_logdir = args.resume_logdir  # None,
    rel_model_pth = args.rel_model_pth #  'checkpoint.pth.tar'

    print('Running setup...')
    cfg = script_utils.get_custom_maskrcnn_cfg(config_filepath)
    head_type = 'custom'
    config_dictionary = {'max_itr': cfg.SOLVER.MAX_ITER, 'head_type': head_type}

    if resume_logdir is not None:
        output_dir = resume_logdir
        assert os.path.exists(os.path.join(output_dir, 'config.yaml'))
        config_outpath = os.path.join(output_dir, "config_resume.yaml")
        with open(config_outpath, "w") as f:
            f.write(cfg.dump())
    else:
        output_dir = local_pyutils.get_log_dir('output/logs/train/train_primary_secondary_full',
                                               config_dictionary=config_dictionary)
        config_outpath = os.path.join(output_dir, "config.yaml")
    with open(config_outpath, "w") as f:
        f.write(cfg.dump())
    print("Full config saved to {}".format(os.path.abspath(config_outpath)))

    checkpoint_resume = None if resume_logdir is None else os.path.join(resume_logdir, rel_model_pth)
    trainer = Trainer_APD(cfg, out_dir=output_dir, interval_validate=1000, n_model_checkpoints=20,
                          checkpoint_resume=checkpoint_resume)

    script_utils.activate_head_type(trainer, head_type)

    print('Beginning training')
    trainer.train()


def get_parser():
    parser = default_argument_parser()
    parser.add_argument('--resume-logdir', required=False, default=None)
    parser.add_argument('--rel-model-pth', required=False, default='checkpoint.pth.tar')
    parser.add_argument('--config_filepath', required=False, default='./detectron2_repo/configs/'
                                                                     'COCO-InstanceSegmentation/'
                                                                     'mask_rcnn_R_50_FPN_3x_APD.yaml')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
