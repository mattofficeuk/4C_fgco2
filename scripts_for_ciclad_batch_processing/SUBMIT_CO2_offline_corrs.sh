#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
queue="days3"

detrends="True False"

seasons="0 1 2 3 4"
seasons="1 2 3 4"

for detrend in $detrends
do
	for season in $seasons
	do
		runscript=${scripts_dir}/wrapper_CO2_offline_corrs.sh

		output=${output_dir}/output_CO2_offline_corrs_${detrend}_${season}.out

		cmd="qsub -q $queue -l mem=5gb -l vmem=6gb $runscript -vdetrend=$detrend,season=$season -j oe -o $output"
		echo $cmd
		$cmd
	done
done
