"""
Script to train a Multimask RCNN model on a dataset
"""
import os
import argparse
import local_pyutils
import torch

# We do this to force the multiroiheads to be added to the registry
from multimaskextension.model import multi_roi_heads_apd
from multimaskextension.data import registryextension

from multimaskextension.train import script_utils
from multimaskextension.train.trainer_apd import Trainer_APD

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def main(config_filepath='./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_APD.yaml',
         resume_logdir=None, rel_model_pth='checkpoint.pth.tar'):
    print('Running setup...')
    cfg = script_utils.get_custom_maskrcnn_cfg(config_filepath)

    head_type = cfg.MODEL.ROI_MASK_HEAD.INIT_ACTIVATED_MASK_HEAD \
        if cfg.MODEL.ROI_HEADS.NAME != 'StandardROIHeads' else None
    config_dictionary = {'max_itr': cfg.SOLVER.MAX_ITER, 'head_type': head_type}
    if head_type == 'custom':
        config_dictionary.update({'match': int(cfg.MODEL.ROI_MASK_HEAD.MATCHING_LOSS)})

    if resume_logdir is not None:
        output_dir = resume_logdir
        assert os.path.exists(os.path.join(output_dir, 'config.yaml'))
        config_outpath = os.path.join(output_dir, "config_resume.yaml")
        with open(config_outpath, "w") as f:
            f.write(cfg.dump())
    else:
        output_dir = local_pyutils.get_log_dir('output/logs/train/train',
                                               config_dictionary=config_dictionary)
        config_outpath = os.path.join(output_dir, "config.yaml")
    with open(config_outpath, "w") as f:
        f.write(cfg.dump())
    print("Full config saved to {}".format(os.path.abspath(config_outpath)))
    checkpoint_resume = None if resume_logdir is None else os.path.join(resume_logdir, rel_model_pth)
    trainer = Trainer_APD(cfg, out_dir=output_dir, interval_validate=1000, n_model_checkpoints=20,
                          checkpoint_resume=checkpoint_resume)
    if not cfg.MODEL.ROI_HEADS.NAME == 'StandardROIHeads':
        script_utils.activate_head_type(trainer, head_type)

    print('Beginning training')
    trainer.train()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-logdir', required=False, default=None)
    parser.add_argument('--rel-model-pth', required=False, default='checkpoint.pth.tar')
    parser.add_argument('--config_filepath', required=False,
                        default='./configs/mask_rcnn_R_50_FPN_3x_APD.yaml')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(config_filepath=args.config_filepath, resume_logdir=args.resume_logdir,
         rel_model_pth=args.rel_model_pth)
