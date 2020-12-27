#!/bin/bash

i=0
for file in ../configs/2020_12_26_d2s/*.yaml; do
    i=$((i+1))
    if [ $i -gt 1 ]; then break; fi 
    echo "$file"
    noparent="${file##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%m-%d-%Y-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-${basenm}"
    echo "redirected output to file starting with:"
    echo $outst
    srun --gres=gpu:4 --job-name="$basenm" -o "$outst-job-%j.out" -p long bash ../train_bash_wrapper.sh --gpus 0,1,2,3 --config $file &
done
