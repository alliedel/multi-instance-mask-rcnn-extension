
for traind in output/logs/train/train_2021-10-07-*; do
	      traindbase="${traind##*/}"
	      rel_model_pth="model_checkpoints"
	      python val.py --trained-logdir "${traind}" --save-all-predictions --overwrite-preds 0 --overwrite-cocoeval 1 --rel-model-pth "${rel_model_pth}"

	      cocoeval="coco_instances_results_pred_masks.json"
	      for predjson in output/logs/test/${traindbase}/*/*/${cocoeval}; do
		  parentdir="$(dirname "$predjson")"
		  outputdir="~/workspace/images/${parentdir}"
		  python scripts/visualize_data_apd.py --source prediction --dataset single_image --predictions-json "${predjson}" --config-file "${traind}/config.yaml" --output-dir "${outputdir}"
		  echo "visualization data in ${outputdir}"
	      done
	      cocoeval="coco_instances_results_agg-pred_masks1_pred_masks2.json"
	      for predjson in output/logs/test/${traindbase}/*/*/${cocoeval}; do
		  parentdir="$(dirname "$predjson")"
		  outputdir="~/workspace/images/${parentdir}"
		  python scripts/visualize_data_apd.py --source prediction --dataset single_image --predictions-json "${predjson}" --config-file "${traind}/config.yaml" --output-dir "${outputdir}"
		  echo "visualization data in ${outputdir}"
	      done
done
