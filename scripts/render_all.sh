#!/bin/bash
# Render the full video in K parallel pieces (alternating GPU1/GPU2), then join them losslessly.
# usage: render_all.sh <run> <K> <out.mp4>
set -e
cd ~/Desktop/RL_handover
RUN=$1; K=${2:-8}; OUT=${3:-renders/handover_linkedin.mp4}
mkdir -p renders/parts_$RUN; rm -f renders/parts_$RUN/*
export JAX_PLATFORMS=cpu MUJOCO_GL=egl
N=$(.venv/bin/python video_aloha.py $RUN --info --out x 2>/dev/null | tail -1 | .venv/bin/python -c "import json,sys;print(json.loads(sys.stdin.read())['frames'])")
echo "total frames: $N"
for i in $(seq 0 $((K-1))); do
  S=$(( N * i / K )); E=$(( N * (i+1) / K )); GPU=$(( 1 + i % 2 ))
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python video_aloha.py $RUN --start $S --end $E --out renders/parts_$RUN/part_$(printf %02d $i).mp4 > renders/parts_$RUN/log_$i.txt 2>&1 &
done
wait
ls renders/parts_$RUN/part_*.mp4 | sed "s/^/file '/; s/$/'/" | sed "s#file 'renders/parts_$RUN/#file '#" > renders/parts_$RUN/list.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i renders/parts_$RUN/list.txt -c copy renders/parts_$RUN/joined.mp4
# silent audio track: some players and upload pipelines expect one
ffmpeg -y -loglevel error -i renders/parts_$RUN/joined.mp4 -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a aac -b:a 96k -shortest -movflags +faststart $OUT
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,r_frame_rate -of compact $OUT
echo RENDER_DONE
