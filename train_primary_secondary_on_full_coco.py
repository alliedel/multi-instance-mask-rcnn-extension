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
print(os.environ['PYTHONPATH'].split(os.pathsep))
import local_pyutils
"""
The goal of train_two_masks_one_image is to overfit one image to two masks per box.  For the two cats - one chair
image, this means each cat box predicts two cat masks, and the chair box predicts on chair.

Step 1a: Modify multimask loss to send two different sets of weights / ground truth to the loss function
Step 1b: Verify with multimask inference to see that the set of weights is learning the primary or secondary mask,
depending on which one we switch between.

"""

from multimaskextension.train import script_utils
from multimaskextension.train.trainer_apd import Trainer_APD

exporter_ = None


def dbprint(*args, **kwargs):
    print(*args, **kwargs)


def main(config_filepath='./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_APD.yaml'):
    print('Running setup...')
    cfg = script_utils.get_custom_maskrcnn_cfg(config_filepath)
    cfg.SOLVER.MAX_ITER = 100000
    head_type = 'custom'
    config_dictionary = {'max_itr': cfg.SOLVER.MAX_ITER, 'head_type': head_type}

    output_dir = local_pyutils.get_log_dir('output/logs/train/train_primary_secondary_full',
                                           config_dictionary=config_dictionary)
    config_outpath = os.path.join(output_dir, "config.yaml")
    with open(config_outpath, "w") as f:
        f.write(cfg.dump())
    print("Full config saved to {}".format(os.path.abspath(config_outpath)))

    trainer = Trainer_APD(cfg, out_dir=output_dir, interval_validate=1000, n_model_checkpoints=20)
    script_utils.activate_head_type(trainer, head_type)

    print('Beginning training')
    trainer.train()


if __name__ == '__main__':
    main()
