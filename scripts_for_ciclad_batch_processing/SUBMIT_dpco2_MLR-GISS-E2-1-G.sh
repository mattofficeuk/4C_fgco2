#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
queue="std"
# queue="day"

models="GISS-E2-1-G GISS-E2-1-G-CC"
blocks="36 45 46 47 48 57 58 59 60 69 70 71"

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
