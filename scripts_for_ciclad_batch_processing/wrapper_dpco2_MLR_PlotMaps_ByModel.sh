#!/usr/bin/env bash

model=$model

python_script=/home/mmenary/python/scripts/dpco2_MLR_PlotMaps_ByModel.py

output=/home/mmenary/python/scripts/output/dpco2_MLR_PlotMaps_ByModel_${model}.txt

/home/mmenary/anaconda2/envs/py3x/bin/python -u $python_script $model > $output 2>&1
