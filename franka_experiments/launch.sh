#!/bin/bash
# usage: launch.sh <name> <gpu> [extra train.py args...]
cd ~/Desktop/RL_handover
name=$1; gpu=$2; shift 2
[ -d runs/$name ] && mv runs/$name runs/${name}_old_$(date +%H%M%S)
mkdir -p runs/$name
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 CUDA_VISIBLE_DEVICES=$gpu nohup setsid .venv/bin/python train.py --name $name "$@" > runs/$name/train.log 2>&1 < /dev/null &
echo "launched $name on GPU $gpu (pid $!)"
