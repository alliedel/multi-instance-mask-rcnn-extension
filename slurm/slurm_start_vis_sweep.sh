#!/bin/bash
# Run this file from the root directory (one level up from here)

i=0

while IFS= read -r line; do
    i=$((i+1))
#    if [ $i -eq 1 ]; then continue; fi 
    echo "$line"  # Name of training output directory
    cfgpth="${line}/config.yaml"
    noparent="${cfgpth##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%Y-%m-%d-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-vis-${basenm}"
    cmd="srun --gres=gpu:1 --job-name=$basenm -o $outst-job-%j.out -p long -t 48:00:00 bash ./vis_bash_wrapper.sh --gpus 0 $line &"
    echo "$cmd"
    eval "$cmd"
    echo "redirected output to file starting with:"
    echo $outst
done < $1
