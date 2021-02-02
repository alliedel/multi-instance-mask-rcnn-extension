#!/bin/bash
# Run this file from the root directory (one level up from here)

i=0

while IFS= read -r line; do
    i=$((i+1))
    if [ $i -gt 1 ]; then break; fi 
    echo "$line"  # Name of training output directory
    cfgpth="${line}/config.yaml"
    modelpth=""#"${line}/checkpoint.pth.tar"
    noparent="${cfgpth##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%m-%d-%Y-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-val-${basenm}"
    cmd="srun --gres=gpu:4 --job-name=$basenm -o $outst-job-%j.out -p long -t 48:00:00 bash ./val_bash_wrapper.sh --gpus 0,1,2,3 --trained-logdir $line &"
    echo "$cmd"
    eval "$cmd"
    echo "redirected output to file starting with:"
    echo $outst
#    srun --gres=gpu:4 --job-name="$basenm" -o "$outst-job-%j.out" -p long -t 48:00:00 bash ./val_bash_wrapper.sh --gpus 0,1,2,3 --model-pth $modelpth --cfg-pth $cfgpth &
done < $1
