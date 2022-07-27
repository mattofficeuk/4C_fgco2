#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
# queue="std"
queue="day"

models="UKESM1-0-LL"
blocks="90 91 92 93 94 102 103 104 105 106 114 115 116 117 118 126 127 128 129"

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
