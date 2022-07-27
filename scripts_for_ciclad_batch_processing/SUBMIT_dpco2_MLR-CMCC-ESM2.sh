#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
queue="std"
# queue="day"

models="CMCC-ESM2"
blocks="84 85 86 87 88 97 98 99 100 101 110 111 112 113 114 123 124 125"

for model in $models
do
	for block in $blocks
	do
	# for block in `seq 0 155`
	# do
		runscript=${scripts_dir}/wrapper_dpco2_MLR.sh

		output=${output_dir}/output_dpco2_MLR_${model}_${block}.out

		cmd="qsub -q $queue -l mem=5gb -l vmem=6gb $runscript -vblock=$block,model=$model -j oe -o $output"
		echo $cmd
		$cmd
	done
done
