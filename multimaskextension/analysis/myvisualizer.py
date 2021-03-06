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

    def get_labels(self, instance_annos):
        classes = [x["category_id"] for x in instance_annos]
        class_names = self.metadata.get("thing_classes", None)
        scores = [x['score'] for x in instance_annos] if 'score' in instance_annos[0] else None
        # Pred version
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
                boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]
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
