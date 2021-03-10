# sudo docker run -p 8503:8503 -it -v /home/allie/data/datasets/d2s/:/d2s/ -v
# /home/allie/afs_directories/espresso/code/multi-instance-mask-rcnn-extension/output/logs/test/:/testdir
# i008/coco_explorer streamlit run coco_explorer.py -- --coco_train /d2s/annotations/D2S_validation.json
# --coco_predictions /testdir/train_primary_secondary_full_2021-01-23-185542_VCS-64d87d7_MAX_ITR-270000_HEAD_TYPE
# -None/d2s_val/itr256000/coco_instances_results_pred_masks.json --images_path /d2s/images/
