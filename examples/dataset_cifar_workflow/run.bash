#!/usr/bin/env bash

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dev

export PYTHONPATH="${PYTHONPATH}:${HOME}/Lab/Learnware/"
echo ${PYTHONPATH}
token="$(date +%s)"
mkdir -p "./log"
echo "The output is redirected to log/${token}.log with token ${token}"

# shellcheck disable=SC2086
nohup python -u main.py prepare_learnware --rebuild=True > "./log/${token}.log" 2>&1 &
echo "With PID = $!"