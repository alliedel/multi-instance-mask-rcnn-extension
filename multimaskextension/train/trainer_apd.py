# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from multimaskextension.data.build import build_detection_train_loader
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import TrainerBase
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from multimaskextension.train.export import ExportConfig, TrainerExporter
from .script_utils import Timer


class PredictorOrTrainerBase_APD(object):
    """
    Modeled after DefaultPredictor in engine.defaults
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model
        """
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions


class Predictor_APD(PredictorOrTrainerBase_APD):
    """
    Modeled after DefaultPredictor in engine.defaults
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.eval()


class Trainer_APD(TrainerBase):
    def __init__(self, cfg, out_dir=None, interval_validate=1000, n_model_checkpoints=20, checkpoint_resume=None,
                 mode='train'):
        super().__init__()
        self.mode = mode
        self.cfg = cfg.clone()  # cfg can be modified by model
        with Timer('Building model'):
            self.model = build_model(self.cfg)
        if self.mode == 'train':
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        else:
            assert self.mode == 'test'
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.iter = 0

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        if self.mode == 'train':
            self.model.train()
            self.optimizer = self.build_optimizer(cfg, self.model)
        else:
            self.model.eval()

        if self.mode == 'train':
            with Timer('Building dataloader'):
                self.train_data_loader = self.build_train_loader(cfg)
            self.data_loader = self.train_data_loader
        else:
            raise NotImplementedError
            self.data_loader = self.val_data_loader
        self._data_loader_iter = iter(self.data_loader)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            self.model = DistributedDataParallel(
                self.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0  # Will be changed if/when resume state is loaded
        self.max_iter = cfg.SOLVER.MAX_ITER if not cfg.GLOBAL.ONE_EPOCH else \
            len(self.data_loader.dataset)  # I should divide by batch size, but not sure how itrs map with distributed
        self.cfg = cfg

        # event storage
        self.storage = None

        metric_makers = {
        }

        export_config = ExportConfig(out_dir, interval_validate=interval_validate,
                                     max_n_saved_models=n_model_checkpoints)
        tensorboard_writer = SummaryWriter(log_dir=out_dir)

        self.exporter = TrainerExporter(
            out_dir=out_dir, export_config=export_config, tensorboard_writer=tensorboard_writer,
            metric_makers=metric_makers)
        self.t_val = None  # We need to initialize this when we make our validation watcher.

        self.load_checkpoint(checkpoint_file=checkpoint_resume)

        self.pbar = tqdm.tqdm(initial=self.start_iter, total=self.max_iter, desc='Training')

    def load_checkpoint(self, checkpoint_file):
        if checkpoint_file is None:
            return
        assert os.path.exists(checkpoint_file)
        state = torch.load(checkpoint_file)
        for key, value in state.items():
            if key == 'best_mean_iu' or key == 'mean_iu':
                pass
            elif key == 'arch':
                assert self.model.__class__.__name__ == value
            elif key == 'epoch':
                pass
            elif key == 'iteration':
                self.start_iter = value
                self.iter = value
            elif key == 'model_state_dict' or key == 'model':
                self.model.load_state_dict(value)
            elif key == 'optim_state_dict':
                self.optimizer.load_state_dict(value)
            else:
                raise NotImplementedError('No loading written for {}'.format(key))

    def train(self):
        """
        Run training.
        Parent calls:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        return super().train(self.start_iter, self.max_iter)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        next_data = next(self._data_loader_iter)
        self.run_step_with_given_data(next_data)

    def run_step_with_given_data(self, data):
        """
        Implement the standard training logic described above.
        """
        self.pbar.update(1)
        memory_allocated = torch.cuda.memory_allocated(device=None)
        self.pbar.set_postfix({'nimgs': len(data), 'mem (GB)': memory_allocated / 1e9})
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        if self.cfg.GLOBAL.NOOP:
            with open('/tmp/imageids.txt', 'a') as f:
                for i in [d['image_id'] for d in data]:
                    f.write(f"{i}\n")
                    print(f"{i}\n")

        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time

        if self.exporter.tensorboard_writer is not None:
            for key, val in loss_dict.items():
                try:
                    val = val.item()
                except AttributeError:
                    try:
                        val = val.sum()
                    except AttributeError:
                        pass

                self.exporter.tensorboard_writer.add_scalar(f"A_trainingloss/{key}",
                                                            val, self.iter)
            # self.exporter.tensorboard_writer.add_scalar(f"B_hyperparams/lr",
            #                                             self.optimizer.state, self.iter)
        if self.iter % self.exporter.export_config.interval_validate == 0:
            if self.exporter.conservative_export_decider.is_prev_or_next_export_iteration(self.iter):
                current_checkpoint_file = self.exporter.save_checkpoint(None, self.iter,
                                                                        self.model, self.optimizer,
                                                                        None, None)

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator
        """
        raise NotImplementedError

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            logger.info('Building test loader')
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            evaluator = (
                evaluators[idx]
                if evaluators is not None
                else cls.build_evaluator(cfg, dataset_name)
            )
            if comm.is_main_process():
                logger.info('Running inference on dataset')
                print('Running inference on dataset')
            results_i, coco_evals = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
                self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                    "iteration", -1
                )
                + 1
        )
