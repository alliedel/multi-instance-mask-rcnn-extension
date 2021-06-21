#!/usr/bin/env python
# coding: utf-8
import os
from glob import glob
import pickle
import pandas as pd
import numpy as np
from pprint import pprint
import yaml
from itertools import chain
import yaml
from bokeh.io import output_file, show, output_notebook
from bokeh.palettes import Spectral5, Turbo256
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import FactorRange
from bokeh.layouts import column, row
from bokeh.io import export_png

from multimaskextension.analysis.bokehutils import grouped_bar, get_blank_figure
from multimaskextension.analysis.pdanalysis import get_datetime, findDiff, get_value_from_cfg, \
    logdirs_to_df, inplace_augment_df_with_cfg_columns
from bokeh.io import show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure

MULTIMASKROOT = "/home/adelgior/afs_directories/espresso/code/multi-instance-mask-rcnn-extension"

if not os.path.exists(MULTIMASKROOT):
    raise Exception(f"Must change MULTIMASKROOT path in {__file__}")


def main():
    # Get log directory list
    logdir_regex = os.path.join(MULTIMASKROOT, 'output/logs/test/train_2021*')
    logdirs = sorted(glob(logdir_regex))
    if len(logdirs) == 0:
        print(f"No log directories available in {logdir_regex}")
        return

    # Create dataframe
    df = logdirs_to_df(logdirs)

    # Add auxiliary columns
    for new_colnm, fnc in {
        'date': lambda x: get_datetime(x.trainname)[0],
        'time': lambda x: get_datetime(x.trainname)[1],
    }.items():
        if new_colnm in df:
            print(f"{new_colnm} already in df")
        df[new_colnm] = df.apply(fnc, axis=1)

    # Reformat existing columns
    for old_colnm, reformat_fcn in {
        'date': pd.to_datetime,
    }.items():
        if old_colnm in df:
            df[old_colnm] = reformat_fcn(df[old_colnm])
    print('Loading configs for comparison')
    augmented_col_to_keys = inplace_augment_df_with_cfg_columns(df)

    # Config adjustment - 'Vanilla' still says active head is custom by default
    if 'MODEL-ROI_MASK_HEAD-INIT_ACTIVATED_MASK_HEAD' in df.columns and \
            'MODEL-ROI_MASK_HEAD-NAME' in df.columns:
        df[df['MODEL-ROI_MASK_HEAD-NAME'] == 'StandardROIHeads',
           'MODEL-ROI_MASK_HEAD-INIT_ACTIVATED_MASK_HEAD'] = None

    # Generate a unique id for each experiment
    unique_id_keys = list(augmented_col_to_keys.keys()) + ['itr'] + ['date'] + \
                     ['task_name'] + ['time']
    unique_id = df.apply(lambda x: tuple(tuple((k, f"{x[k]}") for k in unique_id_keys)), axis=1)
    df['uniqueid'] = unique_id
    uniqueid_as_str = df.uniqueid.apply(lambda x: ', '.join(str(k) for k in x))
    df['uniqueid_as_str'] = uniqueid_as_str

    # Generate a train tag with the configurations
    df['traintag'] = df.apply(
        lambda x: tuple(tuple((colnm, x[colnm]) for colnm in augmented_col_to_keys.keys())),
        axis=1)
    traintag_as_str = df.traintag.apply(lambda x: ', '.join(str(k) for k in x))
    assert len(df.traintag) == len(traintag_as_str)
    df['traintag_as_str'] = traintag_as_str

    custom_visualizations(df)


def custom_visualizations(df_segm):
    filters = {
        # 'MODEL-ROI_MASK_HEAD-MATCHING_LOSS': False,
        # 'SOLVER-BASE_LR': 0.02,
        'task_name': ['segm-pred_masks', 'segm-pred_masks1', 'segm-pred_masks2',
                      'segm-agg-pred_masks1_pred_masks2'],
    }
    for f in filters.keys():
        assert f in df_segm, f"{f} not in df columns"
    df_filt = df_segm
    for nm, val in filters.items():
        if type(val) is list:
            df_filt = df_filt[df_filt[nm].apply(lambda x: x in val)]
        else:
            df_filt = df_filt[df_filt[nm].apply(lambda x: x == val)]
        if len(df_filt) == 0:
            print(f"No matches for {nm} == {val}")

    print('Starting visualizations')
    df_tmp = df_filt
    df_tmp.itr = df_tmp.itr.apply(str)
    output_file("filename.html")
    panels = []
    x_super_name = 'uniqueid'
    x_sub_name = 'itr'
    x_sub_unique = sorted(list(df_tmp[x_sub_name].unique()), key=lambda x: int(x))
    x_super_unique = sorted(list(df_tmp[x_super_name].unique()))
    x = [(x1, x2) for x1     in x_super_unique for x2 in x_sub_unique]
    factor_range = FactorRange(*x)
    palette = palettes.cividis(len(x_sub_unique))
    cmap = palette if len(x_sub_unique) <= len(palette) else Turbo256
    panels = []
    for split in df_tmp.split.unique():
        df_tab = df_tmp
        df_tab = df_tab[df_tab.split == split]
        tab_name = split
        row_ps = []
        for y_name in ['AP']:  # , 'AP50', 'AP75', 'APm', 'APl']:
            col_ps = []
            for taskname in df_tmp.task_name.unique():
                title = f"{taskname}: {x_super_name} {y_name} by {x_sub_name}"
                df_vis = df_tab[df_tab.task_name == taskname]
                if len(df_vis) == 0:
                    p = get_blank_figure(title=title)
                else:
                    colors = [cmap[x_sub_unique.index(x_)] for x_ in df_vis[x_sub_name]]
                    df_vis_as_dict = {
                        x_sub_name: df_vis[x_sub_name].tolist(),
                        x_super_name: df_vis[x_super_name].tolist(),
                        y_name: df_vis[y_name].tolist(),
                        'colors': colors,
                    }
                    x = [(x1, x2) for x1, x2 in
                         zip(df_vis_as_dict[x_super_name], df_vis_as_dict[x_sub_name])]
                    if not len(x) == len(list(set(x))):
                        for vals in list(set(x)):
                            n_instances_of_this_key = sum([i for i, x in enumerate(x) if x == vals])
                            if n_instances_of_this_key > 1:
                                print(n_instances_of_this_key)
                                print('')
                    assert len(x) == len(list(set(x)))
                    p = grouped_bar(df_vis_as_dict, x_super_name=x_super_name,
                                    x_sub_name=x_sub_name,
                                    y_name=y_name, title=title, factor_range=factor_range)
                col_ps.append(p)
            row_ps.append(column(*col_ps))
        panel = Panel(child=row(*row_ps), title=tab_name)
        panels.append(panel)
    #     export_png(panel, filename=f"{tab_name}.png")
    tabs = Tabs(tabs=panels)
    show(tabs)


if __name__ == '__main__':
    main()
