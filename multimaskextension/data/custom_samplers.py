# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import math
from collections import defaultdict
from typing import Optional
import torch
import numpy as np

from torch.utils.data.sampler import Sampler

from detectron2.utils import comm


class GeneralizedRepeatFactorTrainingSampler(Sampler):
    """
    APD: Modified
    Similar to TrainingSampler. In each epoch, an image may appear multiple times based on its
    "repeat factor" (given via file). The GENERALIZED repeat factor for an image is a function of
    the tensor given per image.
    """

    def __init__(self, dataset_dicts, repeat_thresh, repeat_factor_file, shuffle=True, seed=None):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        rep_factors = self._get_repeat_factors(dataset_dicts, repeat_thresh, repeat_factor_file)
        self._int_part = torch.trunc(rep_factors)
        self._frac_part = rep_factors - self._int_part

    def _get_repeat_factors(self, dataset_dicts, repeat_thresh, repeat_factor_file):
        """
        Compute per-image repeat factors.

        Args:
            See __init__.

        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        repeat_freqs = np.load(repeat_factor_file)
        ids_dict = torch.load(repeat_factor_file.replace('.npy', '_meta.pt'))
        image_ids = [d['image_id'] for d in ids_dict]
        assert image_ids == [dataset_dict['image_id'] for dataset_dict in dataset_dicts], \
            'Stats provided in the wrong order for dataset_dict.  Might need to recode to handle ' \
            'this case (reorder), or recompute dataset statistics.'

        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        pseudo_cats_per_img = []
        for dataset_dict, overlaps in zip(dataset_dicts, repeat_freqs):  # For each image (without
            # repeats)
            pseudo_cats = []
            for cat_idx in overlaps.nonzero()[0]:
                n_overlaps = overlaps[cat_idx]
                # 'category' includes number of overlaps
                pseudo_cat = (cat_idx, n_overlaps)
                pseudo_cats.append(pseudo_cat)
                category_freq[pseudo_cat] += 1
            pseudo_cats_per_img.append(pseudo_cats)
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict, pseudo_cats in zip(dataset_dicts, pseudo_cats_per_img):  # For each
            # image (without
            rep_factor = max({category_rep[cat_id] for cat_id in pseudo_cats})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices
