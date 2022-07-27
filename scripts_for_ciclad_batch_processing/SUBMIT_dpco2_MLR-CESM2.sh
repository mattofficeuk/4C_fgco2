#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
# queue="std"
# queue="day"

models="CESM2 CESM2-FV2 CESM2-WACCM CESM2-WACCM-FV2"
blocks="88 89 97 98 99 100 108 109 110 111 119 120 121 122 130 131 132 133 134 140 141 142"

for model in $models
do
	queue="std"
	if [[ $model == "CESM2" ]]
	then
		queue="day"
	fi
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
