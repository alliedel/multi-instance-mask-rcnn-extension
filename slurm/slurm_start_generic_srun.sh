i=$((i+1))
datetime=`date +"%m-%d-%Y-%H-%M-%s"`
outst="/home/adelgior/data/slurm-logs/${datetime}-${basenm}"
echo "redirected output to file starting with:"
echo $outst
srun --gres=gpu:1 --job-name="generic" -o "$outst-job-%j.out" -p long bash ./slurm/generic_wrapper.sh --gpus 0 python scripts/assemble_model_comparison_galleries.py
