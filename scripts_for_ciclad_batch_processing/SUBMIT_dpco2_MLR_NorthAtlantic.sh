#!/usr/bin/env bash

model_sets="CMCC-ESM2 GISS-E2-1-G CESM2 CNRM-ESM2-1 NorESM2-LM EC-Earth3-CC UKESM1-0-LL IPSLCM6A"

for model_set in $model_sets
do
  script="/home/mmenary/python/scripts/SUBMIT_dpco2_MLR-${model_set}.sh"
  bash $script
done
