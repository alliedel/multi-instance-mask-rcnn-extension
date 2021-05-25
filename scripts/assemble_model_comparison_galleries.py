import glob
import os
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
import numpy as np
from skimage.io import imsave
import argparse

from multimaskextension.analysis.gallery_utils import get_text_img, get_mask_row_loc, \
    zeropad_arrlist, get_mask_name_from_coco_instances_file_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-font',
                        default="/home/adelgior/code/multi-instance-mask-rcnn-extension/data"
                                "/cache/vis/gt/d2s/")
    parser.add_argument('--gt-segmvis-dir',
                        default="/home/adelgior/code/multi-instance-mask-rcnn-extension/data"
                                "/cache/vis/gt/d2s/")
    parser.add_argument('--pred-segmvis-dir',
                        default="/home/adelgior/code/multi-instance-mask-rcnn-extension/data"
                                "/cache/vis/pred/")
    parser.add_argument('--gallery-outdir', default="/home/adelgior/workspace/images/d2s_vis/")
    parser.add_argument('--trained-model-types', nargs='+',
                        default=['MATCH-1', 'MATCH-0', 'HEAD_TYPE-standard', 'HEAD_TYPE-None'])
    return parser.parse_args()


def main():
    args = parse_args()
    gt_filenames_dict = {}
    pred_filenames_dict = {}
    for trained_model in glob.glob(os.path.join(args.pred_segmvis_dir, '*')):
        for dataset_pth in glob.glob(os.path.join(trained_model, '*')):
            dataset = os.path.basename(dataset_pth)
            for itr_pth in glob.glob(os.path.join(dataset_pth, '*')):
                for mask_pth in glob.glob(os.path.join(itr_pth, '*')):
                    k = (os.path.basename(trained_model), os.path.basename(dataset_pth),
                         os.path.basename(itr_pth),
                         os.path.basename(mask_pth))
                    pred_filenames_dict[k] = sorted(glob.glob(os.path.join(mask_pth, '*')),
                                                    key=alphanumeric_key)
            gt_filenames_dict[dataset] = [os.path.join(args.gt_segmvis_dir, os.path.basename(f)) for
                                          f in
                                          pred_filenames_dict[k]]

    gt_by_id = {os.path.basename(f).split('.')[0]: f for f in
                sorted(list(gt_filenames_dict['d2s_val']) + list(gt_filenames_dict['d2s_train']),
                       key=alphanumeric_key)}
    pred_by_id = {imid: {} for imid in gt_by_id}
    for k in pred_filenames_dict:
        for f in pred_filenames_dict[tuple(k)]:
            knew = list(k)
            imid = os.path.basename(f).split('.')[0]
            if imid not in pred_by_id:
                pred_by_id[imid] = {}
            knew[0] = [x for x in args.trained_model_types if x in k[0]][0]
            knew.pop(2)
            knew.pop(1)
            pred_by_id[imid][tuple(knew)] = f

    if not os.path.exists(args.gallery_outdir):
        os.makedirs(args.gallery_outdir)

    n_export = 0
    for imid in pred_by_id.keys():
        if imid not in gt_by_id:
            continue
        gt = gt_by_id[imid]
        rows = []
        for modeltype in args.trained_model_types:
            ks = [k for k in pred_by_id[imid].keys() if k[0] == modeltype]
            if 'MATCH' in modeltype:
                ks = [k for k in ks if k[1] != 'coco_instances_results_pred_mask']
            ks = sorted(ks, key=lambda x: get_mask_row_loc(
                get_mask_name_from_coco_instances_file_name(x[1])))
            if len(ks) > 0:
                col_fnames = [pred_by_id[imid][k] for k in ks]
                segm_imgs = [imread(f) for f in col_fnames]
                titles = [f"{modeltype} | {str(get_mask_name_from_coco_instances_file_name(k[1]))}"
                          for k in ks]
                title_imgs = [get_text_img((r.shape[0] // 2, r.shape[1]), str(ttl)) for ttl, r in
                              zip(titles, segm_imgs)]
                row_imgs = [np.concatenate([ttli, i], axis=0) for ttli, i in
                            zip(title_imgs, segm_imgs)]
                row = np.concatenate(row_imgs, axis=1)
                rows.append(row)
        rows = zeropad_arrlist(rows)
        im = np.concatenate(rows, axis=0)
        imname = os.path.join(args.gallery_outdir, f"{imid}.jpg")
        print('Exported ', imname)
        imsave(imname, im)
        n_export += 1


if __name__ == '__main__':
    main()
