#!/bin/bash
# Run this file from the root directory (one level up from here)

if [ -z "$1" ]; then
    echo "Please give a directory or file, e.g. configs/2021-01-27/"
    exit 1
fi

cfglist=(${1})

for f in ${cfglist[@]}; do
    echo $f
done
if [ -n "$short" ]; then
    shortorlong="short"
else
    shortorlong="long"
fi


i=0
for file in ${cfglist[@]}; do
    i=$((i+1))
    echo "Starting job for $file"
    noparent="${file##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%Y-%m-%d-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-${basenm}"
    echo "redirected output to file starting with:"
    echo $outst
    srun --gres=gpu:4 --job-name="$basenm" -o "$outst-job-%j.out" -p "$shortorlong" -t 48:00:00 bash ./train_bash_wrapper.sh --gpus 0,1,2,3 --config "$@" &  # Usually you want this to be --config, --resume, etc.
    sleep 4.0
done
