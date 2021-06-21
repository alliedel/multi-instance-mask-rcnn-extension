import datetime

import numpy as np
import os
import pytz
import shutil
import torch
from tensorboardX import SummaryWriter

from multimaskextension.analysis import visualization_utils

MY_TIMEZONE = 'America/New_York'


class ExportConfig(object):

    def __init__(self, out_dir, interval_validate=None, max_n_saved_models=None):
        self.out_dir = out_dir
        self.interval_validate = interval_validate
        self.max_n_saved_models = 30 if max_n_saved_models is None else max_n_saved_models
        self.export_component_losses = True


class ModelHistorySaver(object):
    def __init__(self, model_checkpoint_dir, interval_validate, max_n_saved_models=20,
                 max_n_iterations=100000):
        assert np.mod(max_n_saved_models, 2) == 0, 'Max_n_saved_models must be even'
        self.model_checkpoint_dir = model_checkpoint_dir
        if not os.path.exists(model_checkpoint_dir):
            raise Exception('{} does not exist'.format(model_checkpoint_dir))
        self.interval_validate = interval_validate
        self.max_n_saved_models = max_n_saved_models
        n_digits = max(6, np.ceil(np.log10(max_n_iterations + 1)))
        self.itr_format = '{:0' + str(n_digits) + 'd}'
        self.adaptive_save_model_every = 1
        self.last_model_saved = None

    def get_list_of_checkpoint_files(self):
        return [os.path.join(self.model_checkpoint_dir, f) for f in
                sorted(os.listdir(self.model_checkpoint_dir))]

    def get_latest_checkpoint_file(self):
        return self.get_list_of_checkpoint_files()[-1]

    def get_model_filename_from_iteration(self, i):
        return os.path.join(self.model_checkpoint_dir,
                            'model_' + self.itr_format.format(i) + '.pth.tar')

    def get_iteration_from_model_filename(self, model_filename):
        itr_as_06d = os.path.basename(model_filename).split('_')[1].split('.')[0]
        assert itr_as_06d.isdigit()
        return int(itr_as_06d)

    def save_model_to_history(self, current_itr, checkpoint_file_src, clean_up_checkpoints=True):
        if np.mod(current_itr, self.adaptive_save_model_every * self.interval_validate) == 0:
            shutil.copyfile(checkpoint_file_src,
                            self.get_model_filename_from_iteration(current_itr))
            self.last_model_saved = self.get_model_filename_from_iteration(current_itr)
            if clean_up_checkpoints:
                self.clean_up_checkpoints()
            return True
        else:
            return False

    def clean_up_checkpoints(self):
        """
        Cleans out history to keep only a small number of models; always ensures we keep the
        first and most recent.
        """
        most_recent_file = self.get_latest_checkpoint_file()
        most_recent_itr = self.get_iteration_from_model_filename(most_recent_file)
        n_vals_so_far = most_recent_itr / self.interval_validate
        if (n_vals_so_far / self.adaptive_save_model_every) >= (self.max_n_saved_models):
            while (n_vals_so_far / self.adaptive_save_model_every) >= self.max_n_saved_models:
                self.adaptive_save_model_every *= 2  # should use ceil, log2 to compute instead (
                # this is hacky)
            iterations_to_keep = list(range(0, most_recent_itr + self.interval_validate,
                                            self.adaptive_save_model_every *
                                            self.interval_validate))
            if most_recent_itr not in iterations_to_keep:
                iterations_to_keep.append(most_recent_itr)
            for j in iterations_to_keep:  # make sure the files we assume exist actually exist
                f = self.get_model_filename_from_iteration(j)
                if not os.path.exists(f):
                    print('WARNING: {} does not exist'.format(f))

            for model_file in self.get_list_of_checkpoint_files():
                iteration_number = self.get_iteration_from_model_filename(model_file)
                if iteration_number not in iterations_to_keep:
                    os.remove(model_file)
            assert len(self.get_list_of_checkpoint_files()) <= (
                    self.max_n_saved_models + 1), 'DebugError'


def log(num_or_vec, base):
    # log_b(x) = log_c(x) / log_c(b)
    return np.log(num_or_vec) / np.log(base)


class ConservativeExportDecider(object):
    def __init__(self, base_interval):
        """
        Decides whether we should export something
        """
        self.base_interval = base_interval
        self.n_previous_exports = 0
        self.power = 2  # 3

    def get_export_iteration_list(self, max_iterations):
        return [(x * self.base_interval) ** 3 for x in
                range(0, np.ceil(log(max_iterations, 3)) + 1)]

    @property
    def next_export_iteration(self):
        return self.get_item_in_sequence(self.n_previous_exports)

    @property
    def current_export_iteration(self):
        return None if self.n_previous_exports == 0 else self.get_item_in_sequence(
            self.n_previous_exports - 1)

    def get_item_in_sequence(self, index):
        return self.base_interval * (index ** self.power)

    def is_prev_or_next_export_iteration(self, iteration):
        if iteration > self.next_export_iteration:
            if self.next_export_iteration == 0:  # Hack for 'resume'
                while iteration > self.get_item_in_sequence(self.n_previous_exports):
                    self.n_previous_exports += 1
            else:
                print(
                    Warning('Missed an export at iteration {}'.format(self.next_export_iteration)))
                while iteration > self.get_item_in_sequence(self.n_previous_exports):
                    self.n_previous_exports += 1

        if iteration == self.next_export_iteration:
            self.n_previous_exports += 1
            return True
        else:
            return iteration == self.current_export_iteration


class TrainerExporter(object):
    log_headers = ['']

    def __init__(self, out_dir, export_config: ExportConfig,
                 tensorboard_writer: SummaryWriter = None, metric_makers=None):

        self.export_config = export_config

        # Copies of things the trainer was given access to

        # Helper objects
        self.tensorboard_writer = tensorboard_writer

        # Log directory / log files
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if not os.path.exists(os.path.join(self.out_dir, 'log.csv')):
            with open(os.path.join(self.out_dir, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        # Logging parameters
        self.timestamp_start = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))

        self.val_losses_stored = []
        self.train_losses_stored = []
        self.joint_train_val_loss_mpl_figure = None  # figure for plotting losses on same plot
        self.iterations_for_losses_stored = []

        self.metric_makers = metric_makers

        # Writing activations

        self.run_loss_updates = True
        model_checkpoint_dir = os.path.join(self.out_dir, 'model_checkpoints')
        if not os.path.exists(model_checkpoint_dir):  # if resume, this will be false
            os.mkdir(model_checkpoint_dir)
        self.model_history_saver = ModelHistorySaver(model_checkpoint_dir=model_checkpoint_dir,
                                                     interval_validate=self.export_config.interval_validate,
                                                     max_n_saved_models=self.export_config.max_n_saved_models)
        self.conservative_export_decider = ConservativeExportDecider(
            base_interval=self.export_config.interval_validate)

    @property
    def instance_problem_path(self):
        return os.path.join(self.out_dir, 'instance_problem_config.yaml')

    def write_eval_metrics(self, eval_metrics, loss, split, epoch, iteration):
        with open(os.path.join(self.out_dir, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone(MY_TIMEZONE)) -
                    self.timestamp_start).total_seconds()
            if split == 'val':
                log = [epoch, iteration] + [''] * 5 + \
                      [loss] + list(eval_metrics) + [elapsed_time]
            elif split == 'train':
                try:
                    eval_metrics_as_list = eval_metrics.tolist()
                except:
                    eval_metrics_as_list = list(eval_metrics)
                log = [epoch, iteration] + [loss] + eval_metrics_as_list + [''] * 5 + [elapsed_time]
            else:
                raise ValueError('split not recognized')
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def save_checkpoint(self, epoch, iteration, model, optimizer, best_mean_iu, mean_iu,
                        out_dir=None):
        out_name = 'checkpoint.pth.tar'
        out_dir = out_dir or os.path.join(self.out_dir)
        checkpoint_file = os.path.join(out_dir, out_name)
        if hasattr(self, 'module'):
            model_state_dict = model.module.state_dict()  # nn.DataParallel
        else:
            model_state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'arch': model.__class__.__name__,
            'optim_state_dict': optimizer.state_dict(),
            'model_state_dict': model_state_dict,
            'best_mean_iu': best_mean_iu,
            'mean_iu': mean_iu
        }, checkpoint_file)

        self.model_history_saver.save_model_to_history(iteration, checkpoint_file,
                                                       clean_up_checkpoints=True)
        return checkpoint_file

    def copy_checkpoint_as_best(self, current_checkpoint_file, out_dir=None,
                                out_name='model_best.pth.tar'):
        out_dir = out_dir or self.out_dir
        best_checkpoint_file = os.path.join(out_dir, out_name)
        shutil.copy(current_checkpoint_file, best_checkpoint_file)
        return best_checkpoint_file

    def export_visualizations(self, visualizations, iteration, basename='val_', tile=True,
                              out_dir=None):
        if visualizations is None:
            return
        out_dir = out_dir or os.path.join(self.out_dir, 'visualization_viz')
        visualization_utils.export_visualizations(visualizations, out_dir, self.tensorboard_writer,
                                                  iteration, basename=basename, tile=tile)

    def run_post_train_iteration(self, loss_dict, iteration):
        """
        get_activations_fcn=self.model.get_activations
        """
        eval_metrics = []
        if self.tensorboard_writer is not None:
            # TODO(allie): Check dimensionality of loss to prevent potential bugs
            self.tensorboard_writer.add_scalar('A_eval_metrics/train_minibatch_loss',
                                               loss_dict['average'], iteration)
        return eval_metrics
