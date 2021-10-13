import os
import torch
import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from multimaskextension.train import script_utils
from multimaskextension.analysis.vis_utils import plt_imshow, FigExporter


testtraindir = '/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/output/logs/test\
/train_primary_secondary_full_2021-01-23-185542_VCS-64d87d7_MATCH-0'
testsplit = 'd2s_val_occlusion'
itr_name = 'itr256000'

testdir = os.path.join(testtraindir, testsplit, itr_name)

# prediction_file = 'output/mask_rcnn_R_50_FPN_3x/predictions/output_{}.pt'.format(image_id)
# proposals_file = 'output/mask_rcnn_R_50_FPN_3x/proposals/proposals_{}.pt'.format(image_id)

preds_pth_name = 'instances_predictions_agg-pred_masks1_pred_masks2.pth'

os.path.join(testdir, coco_instances_results_agg-pred_masks1_pred_masks2.json)

exporter = FigExporter()

predictions = torch.load(prediction_file)
proposals = torch.load(proposals_file)
img = cv2.imread(image_filenames[index])

cfg = script_utils.get_maskrcnn_cfg()
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
scale = 2.0
v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
v._default_font_size = v._default_font_size * 1.5
v = v.draw_instance_predictions(predictions["instances"].to("cpu"))
plt_imshow(v.get_image()[:, :, ::-1])
exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_prediction')

plt_imshow(img[:, :, ::-1])
exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_input')

v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
v._default_font_size = v._default_font_size * 1.5
proposals.pred_boxes = proposals.proposal_boxes
v = v.draw_instance_predictions(proposals.to('cpu'))
plt_imshow(v.get_image()[:, :, ::-1])
exporter.export_gcf(os.path.splitext(os.path.basename(__file__))[0] + '_' + image_id + '_proposals')
