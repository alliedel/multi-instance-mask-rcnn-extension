i=$((i+1))
datetime=`date +"%m-%d-%Y-%H-%M-%s"`
outst="/home/adelgior/data/slurm-logs/${datetime}-${basenm}"
echo "redirected output to file starting with:"
echo $outst
srun --gres=gpu:1 --job-name="generic" -o "$outst-job-%j.out" -p long bash ./slurm/generic_wrapper.sh --gpus 4 python scripts/cache_dataset_statistics.py
