#!/bin/bash
# Run this file from the root directory (one level up from here)
while :; do
    case $1 in
        -s|--single) single=1
        ;;
        -f|--first) first=1
        ;;
        -a|--allbutfirst) allbutfirst=1
        ;;
        *) break
    esac
    shift
done

if [ -z "$1" ]; then
    echo "Please give a directory, e.g. configs/2021-01-27/"
    exit 1
fi

if [ -n "$single" ]; then
    echo "Processing single file:"
    cfglist=(${1})
else
    cfgsweep=(`ls ${1}/*.yaml`)
    if [ -n "$first" ]; then
	cfglist=(${cfgsweep[0]})
	echo "processing first element:"
    elif [ -n "$allbutfirst" ]; then
	cfglist=${cfgsweep[@]:1}
	echo "processing all but first element:"
    else
	echo "processing all files in $1:"
	cfglist=${cfgsweep[@]}
    fi
fi

for f in ${cfglist[@]}; do
    echo $f
done


i=0
for file in ${cfglist[@]}; do
    i=$((i+1))
    echo "Starting job for $file"
    noparent="${file##*/}"
    basenm="${noparent%.yaml}"
    datetime=`date +"%m-%d-%Y-%H-%M-%s"`
    outst="/home/adelgior/data/slurm-logs/${datetime}-${basenm}"
    echo "redirected output to file starting with:"
    echo $outst
    srun --gres=gpu:4 --job-name="$basenm" -o "$outst-job-%j.out" -p long bash ./train_bash_wrapper.sh --gpus 0,1,2,3 --config $file &
done
