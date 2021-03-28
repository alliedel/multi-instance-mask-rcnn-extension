import os
import numpy as np

from detectron2.data import MetadataCatalog

from multimaskextension.data import dataset_statistics
from multimaskextension.train import script_utils
from multimaskextension.train.trainer_apd import Trainer_APD


class DatasetForStats(object):
    def __init__(self, dataset, semantic_class_vals):
        self.dataset = dataset
        self.semantic_class_vals = semantic_class_vals

    def __getitem__(self, idx):
        dataset_dict = self.dataset.__getitem__(idx)
        return dataset_dict

    def __len__(self):
        return len(self.dataset)


def main():
    config_filepath = './configs/coco/cocostats.yaml'
    cfg = script_utils.get_custom_maskrcnn_cfg(config_filepath)
    dataloader_configured_for_stats = Trainer_APD.build_train_loader(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    dataset_configured_for_stats = DatasetForStats(
        dataloader_configured_for_stats.dataset,
        list(metadata.thing_dataset_id_to_contiguous_id.values()))

    coco_stats_cache_dir = 'output/cache/coco_stats'
    if not os.path.exists(coco_stats_cache_dir):
        os.makedirs(coco_stats_cache_dir)
    np.savez(os.path.join(coco_stats_cache_dir, 'metadata.npz'),
             {'metadata': metadata,
              'semantic_class_vals': dataset_configured_for_stats.semantic_class_vals})

    boxmask_intersections_file = os.path.join(coco_stats_cache_dir, 'boxmask_intersecions.npy')
    boxmask_intersections_cache = dataset_statistics.BoxMaskIntersectionsCOCO(
        dataset_configured_for_stats.semantic_class_vals,
        semantic_class_names=metadata.thing_classes, cache_file=boxmask_intersections_file)
    boxmask_intersections_stats = boxmask_intersections_cache.compute_or_retrieve(
        dataset_configured_for_stats)

    instance_count_file = os.path.join(coco_stats_cache_dir, 'instance_counts.npy')
    instance_counts_cache = dataset_statistics.NumberofInstancesPerSemanticClassCOCO(
        dataset_configured_for_stats.semantic_class_vals, cache_file=instance_count_file)
    inst_counts_stats = instance_counts_cache.compute_or_retrieve(dataset_configured_for_stats)

    semantic_pixel_count_file = os.path.join(coco_stats_cache_dir, 'semantic_pixel_counts.npy')
    semantic_class_pixel_counts_cache = dataset_statistics.PixelsPerSemanticClassCOCO(
        dataset_configured_for_stats.semantic_class_vals, cache_file=semantic_pixel_count_file)
    sem_pix_counts_stats = semantic_class_pixel_counts_cache.compute_or_retrieve(
        dataset_configured_for_stats)

    # occlusion_counts_file = os.path.join(coco_stats_cache_dir, 'occlusion_counts.npy')
    # occlusion_counts_cache = dataset_statistics.OcclusionsOfSameClass(range(n_sem_classes),
    #                                                                   cache_file=occlusion_counts_file)


if __name__ == '__main__':
    main()
