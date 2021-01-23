#!/bin/bash
# Run this file from the root directory (one level up from here)

i=0
for file in configs/2020_12_26_d2s/*.yaml; do
    i=$((i+1))
    echo "$file"
    noparent="${file##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%m-%d-%Y-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-val-${basenm}"
    echo "redirected output to file starting with:"
    echo $outst
    
    #srun --gres=gpu:4 --job-name="$basenm" -o "$outst-job-%j.out" -p long bash ./val_bash_wrapper.sh --gpus 0,1,2,3 --model-pth $modelpth --cfg-pth $cfgpth &
done
