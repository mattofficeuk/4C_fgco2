#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
queue="std"

year0s="1850 2100 2350 2600 2850 3100 3350 3600"

methods="True False"

seasons="0 1 2 3 4"
seasons="1 2 3 4"

for year0 in $year0s
do
	for method in $methods
	do
		for season in $seasons
		do
			runscript=${scripts_dir}/wrapper_CO2_offline.sh

			output=${output_dir}/output_CO2_offline_${year0}_${method}.out

			cmd="qsub -q $queue -l mem=5gb -l vmem=6gb $runscript -vyear0=$year0,method=$method,season=$season -j oe -o $output"
			echo $cmd
			$cmd
		done
	done
done
