#!/bin/bash
# Run this file from the root directory (one level up from here)

for file in ./configs/2021_01_23_d2s/*.yaml; do
    echo "$file"
    noparent="${file##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%m-%d-%Y-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-${basenm}"
    echo "redirected output to file starting with:"
    echo $outst
    srun --time 3-0 --gres=gpu:4 --job-name="$basenm" -o "$outst-job-%j.out" -p long bash ./train_bash_wrapper.sh --gpus 0,1,2,3 --config $file &
done
