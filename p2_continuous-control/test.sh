#!/bin/bash
echo "========================================"
echo "Run unittest..."
python3 -m unittest

if [ $? -ne 0 ]; then
    echo "Unittest failed."
else
    echo "Test ok."
fi

echo "========================================"
echo "Run main.py..."

python3 main.py \
    --env Reacher \
    --algorithm DDPG \
    --actor DDPGActor \
    --critic DDPGCritic \
    --train \
    --play \
    --save_dir logs \
    -e 1

if [ $? -ne 0 ]; then
    echo "main.py failed."
else
    echo "main.py ok."
fi