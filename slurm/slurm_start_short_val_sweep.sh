#!/bin/bash
# Run this file from the root directory (one level up from here)

i=0

while IFS= read -r line; do
    i=$((i+1))
    echo "$line"  # Name of training output directory
    cfgpth="${line}/config.yaml"
    modelpth=""#"${line}/checkpoint.pth.tar"
    noparent="${cfgpth##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%Y-%m-%d-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-val-${basenm}"
    echo "redirected output to file starting with:"
    echo $outst
    srun --gres=gpu:4 --job-name=$basenm -o $outst-job-%j.out -p short -t 48:00:00 bash ./val_bash_wrapper.sh --gpus 0,1,2,3 $line &
done < $1
