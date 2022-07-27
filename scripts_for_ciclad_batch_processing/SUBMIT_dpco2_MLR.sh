#!/usr/bin/env bash

scripts_dir=/home/mmenary/python/scripts
output_dir=/home/mmenary/python/scripts/output
queue="std"
queue="day"

models="GISS-E2-1-G GISS-E2-1-G-CC CMCC-ESM2 NorESM2-LM CESM2 CESM2-WACCM CESM2-WACCM-FV2 CESM2-FV2 CNRM-ESM2-1 UKESM1-0-LL EC-Earth3-CC IPSLCM6A"

## blocks in models:
## IPSLCM6A        156
## UKESM1-0-LL     132
## EC-Earth3-CC    130
## CESM2           143
## CESM2-FV2       143
## CESM2-WACCM     143
## CESM2-WACCM-FV2 143
## NorESM2-LM      156
## CNRM-ESM2-1     130
## CMCC-ESM2       130
## GISS-E2-1-G      60
## GISS-E2-1-G-CC   60

for model in $models
do
	# for block in `seq 0 155`
	for block in `seq 59 59`
	do
		runscript=${scripts_dir}/wrapper_dpco2_MLR.sh

		output=${output_dir}/output_dpco2_MLR_${model}_${block}.out

		cmd="qsub -q $queue -l mem=5gb -l vmem=6gb $runscript -vblock=$block,model=$model -j oe -o $output"
		echo $cmd
		$cmd
	done
done
