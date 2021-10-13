
for traind in output/logs/train/train_2021-10-10-*; do
    [ -e "${traind}/model_checkpoints" ] || continue
    traindbase="${traind##*/}"
    rel_model_pth="model_checkpoints"
    cmd="python val.py --trained-logdir ${traind} --save-all-predictions --overwrite-preds 0 --overwrite-cocoeval 0 --rel-model-pth ${rel_model_pth}"
    echo "$cmd"
    eval "$cmd"

    cocoeval="coco_instances_results_pred_masks.json"
    for predjson in output/logs/test/${traindbase}/*/*/${cocoeval}; do
	[ -e "$predjson" ] || continue
	parentdir="$(dirname "$predjson")"
	outputdir="~/workspace/images/${parentdir}"
	cmd="python scripts/visualize_data_apd.py --source prediction --predictions-json ${predjson} --config-file ${traind}/config.yaml --output-dir ${outputdir}"
	echo "$cmd"
	eval "$cmd"
	echo "visualization data in ${outputdir}"
    done
    cocoeval="coco_instances_results_agg-pred_masks1_pred_masks2.json"
    for predjson in output/logs/test/${traindbase}/*/*/${cocoeval}; do
	[ -e "$predjson" ] || continue
	parentdir="$(dirname "$predjson")"
	outputdir="~/workspace/images/${parentdir}"
	cmd="python scripts/visualize_data_apd.py --source prediction --predictions-json ${predjson} --config-file ${traind}/config.yaml --output-dir ${outputdir}"
	echo "$cmd"
	eval "$cmd"
	echo "visualization data in ${outputdir}"
    done
done
