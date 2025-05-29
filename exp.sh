#!/usr/bin/env bash

mkdir -p loss

declare -a MODEL=('TransE' 'RotatE' 'DistMult' 'ComplEx')
declare -a DATASET=('FB15K237' 'NELL-995' 'WN18RR')

for n in 3 5 10
do
	for dataset in "${DATASET[@]}"
	do
		for model in "${MODEL[@]}"
		do
			file="main.py"
			data="fed_data/${dataset}/${dataset}-Fed${n}.pkl"
			name="${dataset}_fed${n}_fed_${model}"
			cmd="python $file --data_path $data --name $name --run_mode FedPE --model ${model} --early_stop_patience 5 --gpu 0"
			echo $cmd
			eval $cmd
		done
	done
done