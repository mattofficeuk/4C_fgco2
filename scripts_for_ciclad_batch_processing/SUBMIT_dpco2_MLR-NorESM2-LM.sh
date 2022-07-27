#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
queue="std"
# queue="day"

models="NorESM2-LM"
blocks="96 97 98 99 100 108 109 110 111 112 120 121 122 123 124 132 133 134 135 136 144 145 146"

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
