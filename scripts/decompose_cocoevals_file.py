import os
from glob import glob
import argparse
import pickle
import pandas as pd
import numpy as np
from pprint import pprint
import torch

import pandas as pd

# Compile stats by category; store imageids in a separate indexing list


def get_df_agg_from_cocoeval(evalImgs, params):
    N = len(params.imgIds)  # number of images
    C = len(params.catIds)
    A = len(params.areaRng)

    def getimcatareaid(imidx, areaidx, catidx):
        return A*N*k + N*j + i

    agg = {'ious_bygt': {},
           'gtids': {},
           'dt_fp': {},
           'gt_missed': {}
          }
    imcatarea_df_fp = []
    imcatarea_df_fn = []
    imcatarea_df_tp = []
    imcatarea_imgId = []
    imcatarea_catId = []
    imcatarea_ninst = []
    imcatarea_areaRng = []
    inst_df_iou = []
    inst_df_gtid = []
    inst_df_catId = []
    inst_df_imgId = []
    inst_df_areaRng = []
    for k, catid in enumerate(params.catIds):
        for j, arng in enumerate(params.areaRng[:1]):
            for i, imgid in enumerate(params.imgIds):
                imcatareaidx = getimcatareaid(i, j, k)
#                 print(i, j, k, imcatareaidx)
                if evalImgs[imcatareaidx] is not None:
                    n_inst = len(evalImgs[imcatareaidx]['gtIds'])
                    assert (evalImgs[imcatareaidx]['image_id'], evalImgs[imcatareaidx]['category_id'], evalImgs[imcatareaidx]['aRng']) == (imgid, catid, arng)
                    inst_df_iou.extend(evalImgs[imcatareaidx]['gtMatchIous'][0, :])
                    inst_df_gtid.extend(evalImgs[imcatareaidx]['gtIds'])
                    inst_df_catId.extend([catid for _ in range(n_inst)])
                    inst_df_imgId.extend([imgid for _ in range(n_inst)])
                    inst_df_areaRng.extend([arng for _ in range(n_inst)])
                    fp = sum(x == 0 for x in evalImgs[imcatareaidx]['dtMatches'][0, :])
                    tp = sum(x != 0 for x in evalImgs[imcatareaidx]['gtMatches'][0, :])
                    fn = sum(x == 0 for x in evalImgs[imcatareaidx]['gtMatches'][0, :])
                    ninst = len(evalImgs[imcatareaidx]['gtIds'])
                else:
                    fp, tp, fn = -1, -1, -1
                    ninst = 0
                imcatarea_df_fp.append(fp)
                imcatarea_df_fn.append(fn)
                imcatarea_df_tp.append(tp)
                imcatarea_imgId.append(imgid)
                imcatarea_catId.append(catid)
                imcatarea_areaRng.append(arng)
                imcatarea_ninst.append(ninst)


    inst_df = pd.DataFrame(
        data={
        'iou': inst_df_iou,
        'gtid': inst_df_gtid,
        'category_id': inst_df_catId,
        'image_id': inst_df_imgId,
        'aRng': inst_df_areaRng
    }, index=None)
    imcatarea_df = pd.DataFrame(
        data={
            'fp': imcatarea_df_fp,
            'tp': imcatarea_df_tp,
            'fn': imcatarea_df_fn,
            'image_id': imcatarea_imgId,
            'n_inst': imcatarea_ninst,
        }, index=None)
    return inst_df, imcatarea_df


def decompose_cocoeval_file(filepath):
    assert os.path.exists(filepath), filepath + ' does not exist'
    print(f"Loading {filepath}")
    d = torch.load(filepath)
    decomp_dir = filepath.replace('.pth', '-decomp/')
    if os.path.exists(decomp_dir):
        print(f"Warning: directory already exists.")
    else:
        os.makedirs(decomp_dir)
    for taskname in d.keys():
        outname = os.path.join(decomp_dir, '-'.join(x for x in taskname) + '.pth')
        print(f"Saving {outname}")
        torch.save(d[taskname], outname)


def main():
    traindir_roots = glob('/home/adelgior/afs_directories/kalman/code/multi-instance-mask-rcnn-extension/output/logs/test/train_2021*')

    # pprint(traindirs)
    traindirs = {}
    for d in traindir_roots:
        # lst = glob(os.path.join(d, 'coco_2017_val', 'itr*'))
        lst = glob(os.path.join(d, 'coco_2017_val', 'itr16000'))
        headtype = [s for s in ['HEAD_TYPE-custom_MATCH-1', 'HEAD_TYPE-custom_MATCH-0', 'HEAD_TYPE-None', 'HEAD_TYPE-standard'] if s in os.path.basename(d)][0]
        iterations = [int(os.path.basename(l).replace('itr', '')) for l in lst]
        for i, l in zip(iterations, lst):
            if not os.path.exists(os.path.join(l, 'cocoevals-decomp/')):
                print('Decomposing the cocoeval file')
                decompose_cocoeval_file(os.path.join(l, 'cocoevals.pth'))
            task_segm_files = glob(os.path.join(l, 'cocoevals-decomp/*-segm.pth'))
            tasks = [os.path.splitext(os.path.basename(t))[0] for t in task_segm_files]
            traindirs[(headtype, i)] = {t: f for t, f in zip(tasks, task_segm_files)}


def main():
    # Load cocoevals and convert to dataframes if not cached.

    traindirs = {}
    traindir_roots = glob(
        '/home/adelgior/code/multi-instance-mask-rcnn-extension/output'
        '/logs/test/train_2021*')
    for d in traindir_roots:
        lst = glob(os.path.join(d, 'coco_2017_val', 'itr*'))
        # lst = glob(os.path.join(d, 'coco_2017_val', 'itr16000'))

        headtype = [s for s in
                    ['HEAD_TYPE-custom_MATCH-1', 'HEAD_TYPE-custom_MATCH-0', 'HEAD_TYPE-None',
                     'HEAD_TYPE-standard'] if s in os.path.basename(d)][0]
        iterations = [int(os.path.basename(l).replace('itr', '')) for l in lst]
        for i, l in zip(iterations, lst):
            if not os.path.exists(os.path.join(l, 'cocoevals-decomp/')):
                print('Decomposing the cocoeval file')
                decompose_cocoeval_file(os.path.join(l, 'cocoevals.pth'))
            task_segm_files = glob(os.path.join(l, 'cocoevals-decomp/*-segm.pth'))
            tasks = [os.path.splitext(os.path.basename(t))[0] for t in task_segm_files]
            traindirs[(headtype, i)] = {t: f for t, f in zip(tasks, task_segm_files)}
    cocoevals = {}
    i = 0
    df_csv_names = {}
    for k, v in traindirs.items():
        print(f"{i}/{len(traindirs)}")
        cocoeval = {}
        for k1, v1 in v.items():
            inst_df_csv_name = v1.replace('.pth', '-df-inst.csv')
            imcatarea_df_csv_name = v1.replace('.pth', '-df-imcatarea.csv')
            df_csv_names[(k, k1)] = {'inst': inst_df_csv_name, 'imcatarea': imcatarea_df_csv_name}

    inst_dfs = {}
    imcatarea_dfs = {}
    bad_keys = []
    for (k, k1), csv_names in df_csv_names.items():
        if not all(os.path.exists(v) for v in csv_names.values()):
            print(f"loading from {traindirs[k][k1]}")
            try:
                cocoeval = torch.load(traindirs[k][k1])
            except Exception as ex:
                bad_keys.append((k, k1))
                print(f"FAILED Loading from {traindirs[k][k1]}")
                print(ex)
                continue
            inst_df_, imcatarea_df_ = get_df_agg_from_cocoeval(cocoeval.evalImgs, cocoeval.params)
            print(inst_df_)
            inst_df_.to_csv(csv_names['inst'])
            imcatarea_df_.to_csv(csv_names['imcatarea'])
        else:
            print('Files already exist:')
            for v in csv_names.values():
                print(v)
        i += 1
    for (k, k1) in bad_keys:
        del df_csv_names[(k, k1)]
    print('Done')


if __name__ == '__main__':
    # args = parse_args()
    main()
