#!/bin/bash
nvidia-smi

if [ "$1" == "--gpus" ]; then
    shift;
    gpus=$1
    shift;
fi
source ~/.bashrc
#export CUDA_VISIBLE_DEVICES=$gpus
activate_virtualenv pytorch4
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
cmd="python train_primary_secondary_on_full_coco.py $@"
echo "$cmd"

$cmd

echo "Slurm job complete"
