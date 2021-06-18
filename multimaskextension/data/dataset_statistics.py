import numpy as np
import torch
import tqdm
import cv2
import abc
import logging
import os, shutil
import random

from multimaskextension.analysis import visualization_utils
from detectron2.structures.masks import BitMasks

try:
    from tabulate import tabulate
except:
    tabulate = None

logger = logging.getLogger(__name__)


class DatasetStatisticCacheInterface(object):
    """
    Stores some statistic about a set of images contained in a Dataset class (e.g. - number of
    instances in the image).
    Inheriting from this handles some of the save/load/caching.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, cache_file=None, override=False):
        self._stat_tensor = None
        self.cache_file = cache_file
        self.override = override

    @abc.abstractmethod
    def labels(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.stat_tensor.shape

    @property
    def n_images(self):
        return self.shape[0]

    @staticmethod
    def load(stats_filename):
        return torch.from_numpy(np.load(stats_filename))

    @staticmethod
    def save(statistics, stats_filename):
        np.save(stats_filename, statistics)

    @property
    def stat_tensor(self):
        if self._stat_tensor is None:
            raise Exception(
                'Statistic has not yet been computed.  Run {}.compute(<dataset>)'.format(
                    self.__class__.__name__))
        return self._stat_tensor

    def compute_or_retrieve(self, dataset):
        if self.cache_file is None:
            logger.info('Computing statistics without cache')
            self._stat_tensor = self._compute(dataset)
        elif self.override or not os.path.exists(self.cache_file):
            logger.info('Computing statistics for file {}'.format(self.cache_file))
            self._stat_tensor = self._compute(dataset)
            self.save(self._stat_tensor, self.cache_file)
        else:
            logger.info('Loading statistics from file {}'.format(self.cache_file))
            self._stat_tensor = self.load(self.cache_file)

    @abc.abstractmethod
    def _compute(self, dataset):
        raise NotImplementedError

    def print_stat_tensor_for_txt_storage(self, stat_tensor):
        print(stat_tensor)

    def pprint_stat_tensor(self, stat_tensor=None, labels=None, with_labels=True):
        if stat_tensor is None:
            stat_tensor = self.stat_tensor
        n_columns = stat_tensor.size(1)
        if with_labels:
            try:
                headings = labels if labels is not None else self.labels
            except NotImplementedError:
                headings = ['{}'.format(x) for x in range(stat_tensor.size(1))]
            assert n_columns == len(headings)
            nested_list = [headings]
        else:
            nested_list = []
        nested_list += stat_tensor.tolist()
        if tabulate is None:
            print(Warning('pretty print only works with tabulate installed'))
            for l in nested_list:
                print('\t'.join(['{}'.format(x) for x in l]))
        else:
            print(tabulate(nested_list))


class PixelsPerSemanticClass(DatasetStatisticCacheInterface):

    def __init__(self, semantic_class_vals, semantic_class_names=None, cache_file=None,
                 override=False):
        super(PixelsPerSemanticClass, self).__init__(cache_file, override)
        self.semantic_class_names = semantic_class_names or ['{}'.format(v) for v in
                                                             semantic_class_vals]
        self.semantic_class_vals = semantic_class_vals

    @property
    def labels(self):
        return self.semantic_class_names

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        semantic_pixel_counts = self.compute_semantic_pixel_counts(dataset,
                                                                   self.semantic_class_vals)
        self._stat_tensor = semantic_pixel_counts
        tensor_size = torch.Size((len(dataset), len(self.semantic_class_vals)))
        assert semantic_pixel_counts.size() == tensor_size, \
            'semantic pixel counts should be a matrix of size {}, not {}'.format(
                tensor_size, semantic_pixel_counts.size())
        return semantic_pixel_counts

    @staticmethod
    def compute_semantic_pixel_counts(dataset, semantic_class_vals):
        semantic_pixel_counts_nested_list = []
        for idx, data_dict in tqdm.tqdm(
                enumerate(dataset), total=len(dataset),
                desc='Running semantic pixel statistics on dataset'.format(dataset), leave=True):
            img_data, (sem_lbl, inst_lbl) = data_dict['image'], \
                                            (data_dict['sem_lbl'], data_dict['inst_lbl'])
            semantic_pixel_counts_nested_list.append([(sem_lbl == sem_val).sum() for sem_val in \
                                                      semantic_class_vals])
        semantic_pixel_counts = torch.IntTensor(semantic_pixel_counts_nested_list)
        return semantic_pixel_counts


class PixelsPerSemanticClassCOCO(PixelsPerSemanticClass):
    @staticmethod
    def compute_semantic_pixel_counts(dataset, semantic_class_vals):
        semantic_pixel_counts = torch.zeros(size=(len(dataset), len(semantic_class_vals)),
                                            dtype=int)
        for idx, data_dict in tqdm.tqdm(
                enumerate(dataset), total=len(dataset),
                desc='Running semantic pixel statistics on dataset'.format(dataset), leave=True):
            polygon_masks = BitMasks.from_polygon_masks(data_dict['instances'].gt_masks,
                                                        height=data_dict['height'],
                                                        width=data_dict['width'])
            n_pix_per_instance = polygon_masks.tensor.sum(dim=[1, 2])
            for n_pix, sem_cls in zip(n_pix_per_instance, data_dict['instances'].gt_classes):
                sem_idx = semantic_class_vals.index(sem_cls)
                semantic_pixel_counts[idx, sem_idx] += n_pix
        return semantic_pixel_counts


class NumberofInstancesPerSemanticClass(DatasetStatisticCacheInterface):
    """
    Computes NxS nparray: For each of N images, contains the number of instances of each of S
    semantic classes
    """

    def __init__(self, semantic_classes, cache_file=None, override=False):
        super(NumberofInstancesPerSemanticClass, self).__init__(cache_file, override)
        self.semantic_classes = semantic_classes

    @property
    def labels(self):
        return self.semantic_classes

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        instance_counts = self.compute_instance_counts(dataset, self.semantic_classes)
        self._stat_tensor = instance_counts
        return instance_counts

    @staticmethod
    def compute_instance_counts(dataset, semantic_classes):
        instance_counts = torch.ones(len(dataset), len(semantic_classes)) * -1
        for idx, data_dict in \
                tqdm.tqdm(enumerate(dataset), total=len(dataset),
                          desc='Running instance statistics on dataset'.format(dataset),
                          leave=True):
            img_data, (sem_lbl, inst_lbl) = data_dict['image'], (data_dict['sem_lbl'], data_dict[
                'inst_lbl'])

            for sem_idx, sem_val in enumerate(semantic_classes):
                sem_locations_bool = sem_lbl == sem_val
                if torch.sum(sem_locations_bool) > 0:
                    my_max = inst_lbl[sem_locations_bool].max()
                    instance_counts[idx, sem_idx] = my_max
                else:
                    instance_counts[idx, sem_idx] = 0
                if sem_idx == 0 and instance_counts[idx, sem_idx] > 0:
                    import ipdb
                    ipdb.set_trace()
                    raise Exception('inst_lbl should be 0 wherever sem_lbl is 0')
        return instance_counts


class NumberofInstancesPerSemanticClassCOCO(NumberofInstancesPerSemanticClass):
    """
    Same as NumberofInstancesPerSemanticClass; computes on COCO format instead.

    Computes NxS nparray: For each of N images, contains the number of instances of each of S
    semantic classes
    """

    @staticmethod
    def compute_instance_counts(dataset, semantic_classes):
        instance_counts = torch.ones(len(dataset), len(semantic_classes)) * -1
        for idx, data_dict in \
                tqdm.tqdm(enumerate(dataset), total=len(dataset),
                          desc='Running instance statistics on dataset'.format(dataset),
                          leave=True):
            for sem_idx, sem_val in enumerate(semantic_classes):
                n_instances = sum(data_dict['instances'].gt_classes == sem_val)
                instance_counts[idx, sem_idx] = n_instances
        return instance_counts


def get_occlusions_hws_from_labels(sem_lbl_np, inst_lbl_np, semantic_class_vals):
    """
    Returns (h,w,S) tensor where S is the number of semantic classes
    """
    # Make into batch form to use more general functions
    sem_lbl_np = sem_lbl_np[:, :, None]
    inst_lbl_np = inst_lbl_np[:, :, None]
    list_of_occlusion_locations_per_cls = []
    n_occlusion_pairings_per_sem_cls = np.zeros(len(semantic_class_vals))
    for sem_idx, sem_val in enumerate(semantic_class_vals):
        n_occlusion_pairings, all_occlusion_locations = \
            OcclusionsOfSameClass.compute_occlusions_from_batch_of_one_semantic_cls(
                sem_lbl_np, inst_lbl_np, sem_val)
        assert all_occlusion_locations.shape[2] == 1
        list_of_occlusion_locations_per_cls.append(all_occlusion_locations[:, :, 0])
        n_occlusion_pairings_per_sem_cls[sem_idx] = n_occlusion_pairings
    return n_occlusion_pairings_per_sem_cls, np.stack(list_of_occlusion_locations_per_cls, axis=2)


class OcclusionsOfSameClass(DatasetStatisticCacheInterface):
    default_compute_batch_sz = 10
    debug_dir = None

    def __init__(self, semantic_class_vals, semantic_class_names=None, cache_file=None,
                 override=False, compute_batch_size=None, debug=False):
        super(OcclusionsOfSameClass, self).__init__(cache_file, override)
        self.semantic_class_names = semantic_class_names or ['{}'.format(v) for v in
                                                             semantic_class_vals]
        self.semantic_class_vals = semantic_class_vals
        self.compute_batch_size = compute_batch_size or self.default_compute_batch_sz
        self.debug = debug

    @property
    def labels(self):
        return [s.replace(' ', '_') for s in self.semantic_class_names]

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        occlusion_counts = self.compute_occlusion_counts(dataset, self.semantic_class_vals,
                                                         self.compute_batch_size)
        self._stat_tensor = occlusion_counts
        assert occlusion_counts.size() == torch.Size((len(dataset), len(self.semantic_class_vals)))
        return occlusion_counts

    @staticmethod
    def dilate(img_hwc, kernel=np.ones((3, 3)), iterations=1, dst=None):
        if dst is None:
            dilated_img = cv2.dilate(img_hwc, kernel=kernel, iterations=iterations)
            if img_hwc.shape[2] == 1:
                dilated_img = np.expand_dims(dilated_img, axis=2)
        else:
            cv2.dilate(img_hwc, kernel=kernel, iterations=iterations, dst=dst)
            if img_hwc.shape[2] == 1:
                np.expand_dims(dst, axis=2)
            dilated_img = None
        return dilated_img

    def compute_occlusion_counts(self, dataset, semantic_class_vals=None, compute_batch_size=None,
                                 debug=None):
        debug = debug if debug is not None else self.debug
        semantic_classes = semantic_class_vals or range(dataset.n_semantic_classes)
        # noinspection PyTypeChecker
        occlusion_counts = torch.zeros((
            len(dataset), len(semantic_class_vals)), dtype=torch.int)
        batch_size = compute_batch_size or self.default_compute_batch_sz
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                 sampler=None, num_workers=4)
        batch_img_idx = 0
        for batch_idx, data_dict in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader),
                desc='Running occlusion statistics on dataset'.format(dataset), leave=True):
            sem_lbl_batch, inst_lbl_batch = data_dict['sem_lbl'], data_dict['inst_lbl']
            batch_sz = sem_lbl_batch.shape[0]
            # Populates occlusion_counts
            batch_occlusion_counts = self.compute_occlusions_from_batch(
                sem_lbl_batch, inst_lbl_batch, semantic_classes, start_img_idx=batch_img_idx,
                debug=debug)
            occlusion_counts[batch_img_idx:(batch_img_idx + batch_sz), :] = torch.from_numpy(
                batch_occlusion_counts)
            batch_img_idx += batch_sz
        return occlusion_counts

    @staticmethod
    def torch_label_batch_to_np_batch_for_dilation(tensor):
        return tensor.numpy().astype(np.uint8).transpose(1, 2, 0)

    def compute_occlusions_from_batch(self, sem_lbl_batch, inst_lbl_batch, semantic_classes,
                                      start_img_idx=None, debug=False):
        batch_sz = sem_lbl_batch.size(0)
        batch_occlusion_counts = np.zeros((batch_sz, len(semantic_classes)), dtype=int)
        # h x w x b (for dilation)
        inst_lbl_np = self.torch_label_batch_to_np_batch_for_dilation(inst_lbl_batch)
        sem_lbl_np = self.torch_label_batch_to_np_batch_for_dilation(sem_lbl_batch)

        for sem_idx in semantic_classes:
            n_occlusion_pairings, occlusion_locs = \
                self.compute_occlusions_from_batch_of_one_semantic_cls(sem_lbl_np,
                                                                       inst_lbl_np, sem_idx)
            batch_occlusion_counts[:, sem_idx] = n_occlusion_pairings
            if debug:
                occlusion_and_sem_cls = self.dilate(occlusion_locs.astype('uint8'),
                                                    iterations=4) + \
                                        (sem_lbl_np == sem_idx).astype('uint8')
                self.export_debug_images_from_batch(
                    occlusion_and_sem_cls, ['occlusion_locations_{}_{}_n_occlusions_{}'.format(
                        start_img_idx + i, self.semantic_class_names[sem_idx],
                        n_occlusion_pairings[i])
                        for i in range(batch_sz)])

        # dilate occlusion locations for visibility
        return batch_occlusion_counts

    @classmethod
    def compute_occlusions_from_batch_of_one_semantic_cls(cls, sem_lbl_np, inst_lbl_np, sem_idx):
        batch_sz = sem_lbl_np.shape[2]
        all_occlusion_locations = np.zeros_like(sem_lbl_np).astype(int)
        n_occlusion_pairings = np.zeros(batch_sz, dtype=int)
        # sem_lbl_np, inst_lbl_np: h x w x b
        if cls.cannot_have_occlusions(sem_lbl_np, inst_lbl_np, sem_idx):
            return n_occlusion_pairings, all_occlusion_locations  # 0, zeros
        semantic_cls_bool = sem_lbl_np == sem_idx
        max_num_instances = inst_lbl_np[semantic_cls_bool].max()
        dilated_instance_masks = []
        possible_instance_values = range(1, max_num_instances + 1)

        # Collect all the instance masks
        for inst_val in possible_instance_values:
            # noinspection PyTypeChecker
            inst_loc_bool = cls.intersect_two_binary_masks(semantic_cls_bool,
                                                           (inst_lbl_np == inst_val))
            if not np.any(inst_loc_bool):
                continue
            else:
                dilated_instance_masks.append(cls.dilate(inst_loc_bool.astype(np.uint8),
                                                         kernel=np.ones((3, 3)), iterations=1))
        # Compute pairwise occlusions
        for dilate_idx1 in range(len(dilated_instance_masks)):
            for dilate_idx2 in range(dilate_idx1 + 1, len(dilated_instance_masks)):
                mask_pair_intersection = cls.intersect_two_binary_masks(
                    dilated_instance_masks[dilate_idx1], dilated_instance_masks[dilate_idx2])
                all_occlusion_locations += mask_pair_intersection
                #  NOTE(allie): Below is the computationally expensive line.
                n_occlusion_pairings += (np.any(mask_pair_intersection, axis=(0, 1))).astype(int)

        # n_occlusion_pairings: (b,) ,  all_occlusion_locations: (h,w,b)
        return n_occlusion_pairings, all_occlusion_locations

    @staticmethod
    def clear_and_create_dir(dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

    def prep_debug_dir(self):
        if self.debug_dir is None:
            self.debug_dir = '/tmp/occlusion_debug/'
            self.clear_and_create_dir(self.debug_dir)

    @staticmethod
    def export_label_image(img, filename):
        visualization_utils.write_label(filename, img)

    def export_debug_image(self, img, basename):
        filename = os.path.join(self.debug_dir, '{}.png'.format(basename))
        self.export_label_image(img, filename)

    def export_debug_images_from_batch(self, imgs_as_batch, basenames):
        self.prep_debug_dir()
        batch_sz = imgs_as_batch.shape[2]
        assert len(basenames) == batch_sz
        for img_idx in range(batch_sz):
            self.export_debug_image(imgs_as_batch[:, :, img_idx], basenames[img_idx])

    @staticmethod
    def cannot_have_occlusions(sem_lbl_batch, inst_lbl_batch, sem_idx):
        # Check if it even contains this semantic class
        if (sem_lbl_batch == sem_idx).sum() == 0:
            return True
        # Check if it contains at least two instances
        if inst_lbl_batch[sem_lbl_batch == sem_idx].max() < 2:
            return True
        return False

    @staticmethod
    def intersect_two_binary_masks(mask1: np.ndarray, mask2: np.ndarray):
        # would check that only int values are in (0,1), but too much computation.
        for mask in [mask1, mask2]:
            assert mask.dtype in [np.bool, np.int, np.uint8], \
                'I didnt expect a boolean mask with dtype {}'.format(mask1.dtype)
        return mask1 * mask2


class BoxMaskIntersectionsCOCO(DatasetStatisticCacheInterface):
    mask_side_len = 28

    def __init__(self, semantic_class_vals, semantic_class_names=None, cache_file=None,
                 override=False):
        super().__init__(cache_file, override)
        self.metadata_cache_pt_file = cache_file.replace('.npy', '_meta.pt')
        self.semantic_class_names = semantic_class_names or ['{}'.format(v) for v in
                                                             semantic_class_vals]
        self.semantic_class_vals = semantic_class_vals
        self.semantic_class_names_by_val = {k: v for k, v in zip(self.semantic_class_vals,
                                                                 self.semantic_class_names)}

    def _compute(self, dataset):
        if not os.path.exists(self.metadata_cache_pt_file) or self.override:
            lst_dict_box_mask_intersections_per_inst = \
                self.compute_per_instance_overlap_details(dataset)
            torch.save(lst_dict_box_mask_intersections_per_inst, self.metadata_cache_pt_file)
        detailed_counts_dict = torch.load(self.metadata_cache_pt_file)
        self.assert_image_ids_match(dataset, detailed_counts_dict)
        intersections_per_sem_cls_box_mask = self.derive_stats_tensor_from_detailed_stats(
            detailed_counts_dict) # 3D : [img_idx, box_sem_cls_idx, mask_sem_cls_idx]
        box_mask_intersections_per_sem_cls = torch.diagonal(intersections_per_sem_cls_box_mask,
                                                            dim1=1, dim2=2)
        assert box_mask_intersections_per_sem_cls.size() == torch.Size(
            (len(dataset), len(self.semantic_class_vals)))
        self._stat_tensor = box_mask_intersections_per_sem_cls
        return box_mask_intersections_per_sem_cls

    def assert_image_ids_match(self, dataset, detailed_counts_dict_lst):
        print('Asserting image id matches (turn this off if taking too long)')
        assert len(dataset) == len(detailed_counts_dict_lst)
        all_match = True
        idxs = random.sample(range(len(dataset)), min(len(dataset), 100))
        for i in tqdm.tqdm(idxs, total=len(idxs), desc=f"Checking {len(idxs)}/{len(dataset)} " \
                "image ids to make sure they match"):
            if dataset[i]['image_id'] != detailed_counts_dict_lst[i]['image_id']:
                all_match = False
                break
        if not all_match:
            raise Exception(f"image id sets differ between dataset "
                            f"and those saved in {self.metadata_cache_json_file}! (may be in "
                            f"different order(?)")
        else:
            print(f"All {len(idxs)} random datapoints matched!")

    def derive_stats_tensor_from_detailed_stats(self, detailed_counts_dict_lst,
                                                thresh_box_pixels=8):
        """
        Convert list of dicts (one dict per image) into tensor of size NxSxS,
        where N = # images, S = # semantic classes.
        mask_overlaps_per_sem_cls_box_mask[i,j,k] = # overlaps within image i between a box of
        semantic class j and a mask of semantic class k.
        """
        n_images = len(detailed_counts_dict_lst)
        mask_overlaps_per_sem_cls_box_mask = torch.zeros((n_images, len(self.semantic_class_vals),
                                                          len(self.semantic_class_vals)),
                                                         dtype=torch.int)
        for img_idx, d in tqdm.tqdm(enumerate(detailed_counts_dict_lst), total=len(
                detailed_counts_dict_lst), desc='Turning pre-computed box-mask intersection '
                                                'metadata into a '
                                                'stats tensor to summarize each sem_cls, '
                                                'img_idx in the dataset'):
            overlaps_per_inst = d['boxmask_areas_as_pixels_in_masksidelen'] >= thresh_box_pixels
            for i, box_sem_val in enumerate(self.semantic_class_vals):
                box_inst_bool = d['box_classes'] == box_sem_val
                if box_inst_bool.sum() == 0:  # no boxes of this class
                    continue
                for j, mask_sem_val in enumerate(self.semantic_class_vals):
                    mask_inst_bool = d['box_classes'] == mask_sem_val
                    if mask_inst_bool.sum() == 0:
                        continue  # no masks of this class
                    n_overlaps = overlaps_per_inst[box_inst_bool, :][:, mask_inst_bool].sum()
                    mask_overlaps_per_sem_cls_box_mask[img_idx, i, j] = n_overlaps
        return mask_overlaps_per_sem_cls_box_mask

    def compute_per_instance_overlap_details(self, dataset):
        lst_dict_box_mask_intersections_per_inst = []
        for idx, data_dict in \
                tqdm.tqdm(enumerate(dataset), total=len(dataset),
                          desc='Running instance overlap statistics on dataset'.format(dataset),
                          leave=True):
            instances = data_dict['instances']
            n_instances = len(instances)
            # n_boxes x n_masks (same #)
            n_other_mask_pixels_in_boxes = torch.zeros((n_instances, n_instances), dtype=torch.int)
            full_img_box_repeat = torch.zeros((n_instances, 4))
            full_img_box_repeat[:, 2] = instances.image_size[0]  # d['height']
            full_img_box_repeat[:, 3] = instances.image_size[1]  # d['width']
            for box_idx in range(n_instances):
                masks = instances.gt_masks
                box = instances.gt_boxes[box_idx].tensor
                box_repeat = box.expand(len(masks), -1)
                masks_in_box = masks.crop_and_resize(box_repeat, self.mask_side_len)
                n_other_mask_pixels_in_boxes[box_idx, :] = torch.IntTensor(
                    [m.sum().item() for m in masks_in_box])

            box_areas = ((instances.gt_boxes.tensor[:, 2] - instances.gt_boxes.tensor[:, 0]) *
                         (instances.gt_boxes.tensor[:, 3] - instances.gt_boxes.tensor[:, 1])).T
            mask_side_len = self.mask_side_len
            areas_dict = self.get_interpretable_areas(box_areas, n_other_mask_pixels_in_boxes,
                                                      self.mask_side_len)
            img_info_dict = {
                'image_id': data_dict['image_id'],
                'boxes': instances.gt_boxes.tensor,
                'box_classes': instances.gt_classes,
                'box_class_names': [self.semantic_class_names_by_val[sem_val.item()]
                                    for sem_val in instances.gt_classes],
                'mask_side_len': mask_side_len
            }
            img_info_dict.update(areas_dict)
            lst_dict_box_mask_intersections_per_inst.append(img_info_dict)
            if idx == 1:
                # Break early if we can't save the result
                torch.save(lst_dict_box_mask_intersections_per_inst, '/tmp/tmp.pt')

        return lst_dict_box_mask_intersections_per_inst

    @staticmethod
    def get_interpretable_areas(box_areas, n_other_mask_pixels_in_boxes, mask_side_len):
        assert n_other_mask_pixels_in_boxes.shape[0] == n_other_mask_pixels_in_boxes.shape[1] == \
               len(box_areas)
        # n_instances = len(box_areas)
        # boxmask_areas_as_frac_of_box = n_other_mask_pixels_in_boxes.float() / (mask_side_len ** 2)
        # assert boxmask_areas_as_frac_of_box.max() <= 1 + 1e-7, 'Debug assert'
        # broadcasting rules: box area is same per row
        # boxmask_areas_as_fullimg_pixels = (boxmask_areas_as_frac_of_box.T * box_areas).T
        # mask_areas_as_fullimg_pixels = torch.diagonal(boxmask_areas_as_fullimg_pixels)
        # For each box (row), find out what portion of another object's mask is in the box by
        # dividing all mask intersection values by the corresponding mask size in its original box
        # (e.g. - boxmask_areas_as_fraction_of_full_mask[0, :] =
        #       boxmask_areas_as_full_img_pixels[0, :] / mask_areas_as_full_img_pixels
        # boxmask_areas_as_fraction_of_full_mask = (
        #         boxmask_areas_as_fullimg_pixels / mask_areas_as_fullimg_pixels)
        # assert boxmask_areas_as_frac_of_box.max() <= (1 + 1e-7), boxmask_areas_as_frac_of_box.max()
        # assert boxmask_areas_as_fraction_of_full_mask.max() <= (1.1), \
        #     boxmask_areas_as_fraction_of_full_mask.max()
        return {'box_areas_in_pixels': box_areas,
                'boxmask_areas_as_pixels_in_masksidelen': n_other_mask_pixels_in_boxes,
                # 'boxmask_areas_as_frac_of_box': boxmask_areas_as_frac_of_box,
                # 'mask_areas_as_fullimg_pixels': mask_areas_as_fullimg_pixels,
                # 'boxmask_areas_as_fraction_of_full_mask':
                #     boxmask_areas_as_fraction_of_full_mask,
                # 'mask_side_len': mask_side_len
                }


def get_instance_sizes(sem_lbl, inst_lbl, sem_val, void_vals=(255, -1)):
    bool_sem_cls = sem_lbl == sem_val
    if bool_sem_cls.sum() == 0:
        return [], []
    gt_inst_vals = [x.detach().item() for x in torch.unique(inst_lbl[bool_sem_cls])]
    gt_inst_vals = [x for x in gt_inst_vals if x not in void_vals]
    gt_inst_sizes = []
    for inst_val in gt_inst_vals:
        inst_size = torch.sum(inst_lbl[bool_sem_cls] == inst_val)
        if inst_size == 0:
            raise Exception('Debug error..')
        gt_inst_sizes.append(inst_size)
    return gt_inst_vals, gt_inst_sizes
