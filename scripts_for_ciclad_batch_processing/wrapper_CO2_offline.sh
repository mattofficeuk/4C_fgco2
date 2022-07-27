#!/usr/bin/env bash

year0=$year0
method=$method
season=$season

python_script=/home/mmenary/python/scripts/CO2_offline.py
python_script=/home/mmenary/python/scripts/CO2_offline-DEL.py

output=/home/mmenary/python/scripts/output/output_CO2_offline_${year0}_${method}_${season}.txt

/home/mmenary/anaconda2/envs/py3x/bin/python -u $python_script $year0 $method $season > $output 2>&1
