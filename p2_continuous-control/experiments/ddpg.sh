#!/bin/bash
echo "========================================"
echo "Run ddpg training..."

python3 main.py \
    --env Reacher \
    --algorithm DDPG \
    --actor DDPGActor \
    --critic DDPGCritic \
    --train \
    --eval \
    --n_eval_episode 100 \
    --save_dir ddpg \
    -e 20000
