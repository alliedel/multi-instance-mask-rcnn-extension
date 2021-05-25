import cv2
import numpy as np

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels


class MyVisualizer(Visualizer):
    """
    Allie made this class to handle solely instance segmentation and be able to load
    predictions and gts in the same COCO json format and easily visualize.
    """

    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (MetadataCatalog): image metadata.
        """
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def get_labels(self, instance_annos, annos_use_contiguous=False):
        """
        The annotations should not use the contiguous ids (they should output the thing class
        label in non-contiguous form, rather than the convenient remapping the metadata
        provides).  But if they do (directly from the model, for instance), we'll index class
        names that way.
        """
        classes = [x["category_id"] for x in instance_annos]
        scores = [x['score'] for x in instance_annos] if 'score' in instance_annos[0] else None
        # Pred version
        thing_class_names = self.metadata.get("thing_classes", None)
        if annos_use_contiguous:
            thing_class_ids = list(self.metadata.get("thing_dataset_id_to_contiguous_id",
                                                     None).values())
        else:
            thing_class_ids = list(self.metadata.get("thing_dataset_id_to_contiguous_id",
                                                   None).keys())

        unique_pred_labels = sorted(list(np.unique(classes)))
        assert all(x in thing_class_ids for x in unique_pred_labels),\
            (thing_class_ids, unique_pred_labels)

        class_names = {
            i: v
            for i, v in zip(thing_class_ids, thing_class_names)}
        labels = _create_text_labels(classes, scores, class_names)
        return labels

    def get_colors(self, classes):
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5
        return alpha, colors

    def draw_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None
            if 'bbox_mode' in annos[0]:
                boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in
                         annos]
            else:
                boxes = [x["bbox"] for x in annos]
            labels = self.get_labels(annos)
            alpha, colors = self.get_colors([x['category_id'] for x in annos])

            if self._instance_mode == ColorMode.IMAGE_BW:
                raise NotImplementedError
            #     self.output.img = self._create_grayscale_image(
            #         (predictions.pred_masks.any(dim=0) > 0).numpy()
            #     )
            #     alpha = 0.3

            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts,
                                   assigned_colors=colors, alpha=alpha)
        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = cv2.imread(dic["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output

    @staticmethod
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
            assert image_id in predictions_as_dicts, ValueError(
                f"predictions didn\'t match the file info from the "
                f"dataset ({image_id} in predictions)")
            if 'bbox_mode' not in p:
                p['bbox_mode'] = assumed_bbox_mode
            predictions_as_dicts[image_id]['annotations'].append(p)
        return list(predictions_as_dicts.values())