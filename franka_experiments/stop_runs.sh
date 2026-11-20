#!/bin/bash
# stop every running train.py of this project
for p in $(pgrep -f "RL_handover/.venv/bin/python train.py|^.venv/bin/python train.py"); do kill $p; done
sleep 3; pgrep -af "bin/python train.py" || echo "no training running"
