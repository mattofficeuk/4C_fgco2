#!/usr/bin/env bash

detrend=$detrend
season=$season

python_script=/home/mmenary/python/scripts/CO2_offline_corrs.py

output=/home/mmenary/python/scripts/output/output_CO2_offline_corrs_${detrend}_${season}.txt

/home/mmenary/anaconda2/envs/py3x/bin/python -u $python_script $detrend $season > $output 2>&1
