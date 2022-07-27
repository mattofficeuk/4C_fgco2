#!/usr/bin/env bash

block=$block
model=$model

python_script=/home/mmenary/python/scripts/dpco2_MLR.py
# python_script=/home/mmenary/python/scripts/dpco2_MLR-DEL.py

output=/home/mmenary/python/scripts/output/dpco2_MLR_${model}_${block}.txt

/home/mmenary/anaconda2/envs/py3x/bin/python -u $python_script $block $model > $output 2>&1
