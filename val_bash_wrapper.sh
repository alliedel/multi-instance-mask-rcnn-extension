#!/bin/bash
nvidia-smi

if [ "$1" = "--gpus" ]; then
    shift;
    gpus=$1
    shift;
fi
#source ~/.bashrc
. ~/.bashrc
#export CUDA_VISIBLE_DEVICES=$gpus
activate_virtualenv pytorch4
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
cmd="python val.py $@"
echo "$cmd"

$cmd

echo "Slurm job complete"
