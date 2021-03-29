#python val.py --trained-logdir output/logs/train/train_2021-03-27-232834_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-0 --rel-model-pth model_checkpoints --val-dataset coco_2017_val
python val.py --trained-logdir output/logs/train/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1 --rel-model-pth model_checkpoints --val-dataset coco_2017_val
python val.py --trained-logdir output/logs/train/train_2021-03-27-232834_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-0 --rel-model-pth model_checkpoints --val-dataset coco_2017_train
python val.py --trained-logdir output/logs/train/train_2021-03-27-232835_VCS-021a9a3_MAX_ITR-500000_HEAD_TYPE-custom_MATCH-1 --rel-model-pth model_checkpoints --val-dataset coco_2017_train
