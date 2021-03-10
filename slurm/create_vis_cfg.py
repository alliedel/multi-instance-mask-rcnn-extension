import os, glob, argparse

dirname = os.path.abspath('.')

TESTBASEDIR='output/logs/test/'
VIS_PRED_OUT='data/cache/vis/pred'
VIS_GT_OUT='data/cache/vis/gt/d2s'


def main(outfile, overwrite):
    f = None if outfile is None else open(outfile, 'w')
    testdirs = sorted(os.path.relpath(x, dirname) for x in glob.glob(os.path.join(dirname, TESTBASEDIR, '*/*/')))
    for testdir in testdirs:
        predsdir = os.path.join(testdir, 'itr256000')
        dataset = os.path.basename(testdir)
        for mask_json in glob.glob(os.path.join(predsdir, 'coco_instances_results_*pred*.json')):
            preds_rel_dir = os.path.relpath(mask_json, 'output/logs/test/').rstrip('.json')
            vis_outdir = os.path.join(VIS_PRED_OUT, preds_rel_dir)
            if os.path.exists(vis_outdir) and not overwrite:
                print(f"#### {vis_outdir} already exists")
                continue
            cfg_file = os.path.join(testdir, 'config.yaml')
            if not os.path.exists(cfg_file):
                traindir = 'output/logs/train/'
                trainbasenm = os.path.dirname(os.path.relpath(testdir, TESTBASEDIR))
                cfg_file = os.path.join(traindir, trainbasenm, 'config.yaml')
                assert os.path.exists(cfg_file)
            outstr = f"--source prediction --predictions-json {mask_json} --dataset {dataset} --config-file {cfg_file} --output-dir {vis_outdir}"
            if f is None:
                print(outstr)
            else:
                f.write(outstr + '\n')
                

    # GT
    for dataset in ["d2s_val", "d2s_train"]:
        vis_outdir = VIS_GT_OUT
        outstr = f"--source annotation --dataset {dataset} --config-file configs/2021_01_23_d2s/custom_match.yaml --source annotation --output-dir {vis_outdir}"
        if f is None:
            print(outstr)
        else:
            f.write(outstr + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', default=None)
    parser.add_argument('--overwrite', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.output_file, args.overwrite)
