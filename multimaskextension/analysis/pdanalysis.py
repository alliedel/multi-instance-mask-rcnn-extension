import pickle
from glob import glob
from itertools import chain
import os
import pandas as pd
import yaml

# ('MODEL', 'WEIGHTS') is relevant for training (pretrained weights), but for validation it just
# points to our trained model (will be unique for every cfg, and not a useful cfg to save).
VALCFGLOADBLOCKLIST = (('MODEL', 'WEIGHTS'),)


def get_datetime(traindir):
    """
    Extract the date and time from a training directory name
    """
    date, time = None, None
    for p in traindir.split('_'):
        splt = p.split('-')
        if splt[0] == '2021':
            date = '-'.join(splt[:3])
            time = '-'.join(splt[3:])
    return date, time


def findDiff(d1, d2, stack):
    """
    Compare two dictionaries and return the keys that are different.
    """
    for k, v in d1.items():
        if k not in d2:
            yield stack + ([k] if k else [])
        if isinstance(v, dict):
            for c in findDiff(d1[k], d2[k], [k]):
                yield stack + c
        else:  # leaf
            if d1[k] != d2[k]:
                yield stack + [k]


def get_value_from_cfg(cfg, klst):
    """
    Get value from a nested dictionary by following list of (sub)keys
    """
    cfgk = cfg
    for subk in klst:
        cfgk = cfgk[subk]
    return cfgk


def inplace_augment_df_with_cfg_columns(df, auto_diffcfg_columns=True, manual_cfg_allowdict=None,
                                        cf_name='config_file', blocklist=VALCFGLOADBLOCKLIST):
    """
    Populates df with columns derived from config values (will be loaded from the config file)
    auto_diffcfg_columns: Assigns values to any keys that don't match between cfg dictionaries
        (note this means the behavior for each traindir entry depends on the cfgs of other
        traindirs given to this function)
    manual_cfg_allowdict: dict(k1=v1, k2=v2, ...) where k1 is the resulting column name and v1 is
        the nested cfg key list of the value that should be loaded in there.
        e.g.: dict(repeat_threshold=cfg['DATALOADER']['REPEAT_THRESHOLD'], ...)
    cf_name: Column name in df that specifies the config file path
    """
    unique_cfgs = list(set(list(df[cf_name])))
    all_cfgs = {config_file: yaml.load(open(config_file, 'rb')) for config_file in df[cf_name]}

    if auto_diffcfg_columns:
        all_diffs = []
        for i, ci in enumerate(all_cfgs.values()):
            for j, cj in enumerate(all_cfgs.values()):
                if i == j:
                    continue
                all_diffs.append(list(findDiff(ci, cj, [])))

        all_unique_diffs = list(set(chain.from_iterable([tuple(x) for x in diffs]
                                                        for diffs in all_diffs)))
        all_nested_keys_diff = {'-'.join(k): k for k in sorted(all_unique_diffs)}
        cfg_cols_to_keys = {k: v for k, v in all_nested_keys_diff.items() if v not in blocklist}
    else:
        cfg_cols_to_keys = {}
    if manual_cfg_allowdict is not None:
        for colnm, cfgkey in manual_cfg_allowdict.items():
            if colnm in cfg_cols_to_keys.keys():
                assert cfg_cols_to_keys[colnm] == cfgkey
            else:
                cfg_cols_to_keys[colnm] = cfgkey

    for col_name, cfg_key in cfg_cols_to_keys.items():
        def f(x):  # linter doesn't want this as lambda for some reason
            return get_value_from_cfg(all_cfgs[x], cfg_key)

        df[col_name] = df[cf_name].apply(f)

    return cfg_cols_to_keys


def logdirs_to_df(logdirs, segm_task_only=True):
    """
    """
    pd_dicts = []
    for d in logdirs:
        trainname = os.path.basename(d)
        for val_split in [os.path.basename(x) for x in glob(os.path.join(d, '*'))]:
            itr_dirs = glob(os.path.join(d, val_split, 'itr*'))
            for itr_dir in itr_dirs:
                config_file = os.path.join(itr_dir, 'config_resume.yaml')
                itr = int(os.path.basename(itr_dir).replace('itr', ''))
                res_pkl_file = os.path.join(itr_dir, 'results.pkl')
                if not os.path.exists(res_pkl_file):
                    print(f"Warning: {res_pkl_file} does not exist")
                    continue
                print(f"Reading {res_pkl_file}")
                res = pickle.load(open(res_pkl_file, 'rb'))
                for task_name in res.keys():
                    if segm_task_only and 'segm' not in task_name:
                        continue
                    stats_d = {}
                    for stat_name in res[task_name].keys():
                        stats_d[stat_name] = res[task_name][stat_name]
                    entry = {'trainname': trainname,
                             'logdir': d,
                             'config_file': config_file,
                             'task_name': task_name,
                             'itr': itr,
                             'split': val_split}
                    entry.update(stats_d)
                    pd_dicts.append(entry)

    df = pd.DataFrame(pd_dicts)
    return df
