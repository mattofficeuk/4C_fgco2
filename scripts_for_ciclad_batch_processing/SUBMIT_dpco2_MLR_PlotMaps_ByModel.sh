#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
queue="std"

models="IPSLCM6A GISS-E2-1-G GISS-E2-1-G-CC CMCC-ESM2 NorESM2-LM CESM2 CESM2-WACCM CESM2-WACCM-FV2 CESM2-FV2 CNRM-ESM2-1 UKESM1-0-LL EC-Earth3-CC"

## Does this work??
lines=$(qstat -u mmenary | wc -l)
while (( $lines > 5 ))
do
	echo "Waiting for 1 hour"
	sleep 1h
	lines=$(qstat -u mmenary | wc -l)
done

for model in $models
do
	runscript=${scripts_dir}/wrapper_dpco2_MLR_PlotMaps_ByModel.sh

	output=${output_dir}/output_dpco2_MLR_PlotMaps_ByModel_${model}.out

	cmd="qsub -q $queue -l mem=5gb -l vmem=6gb $runscript -vmodel=$model -j oe -o $output"
	echo $cmd
	$cmd
done
