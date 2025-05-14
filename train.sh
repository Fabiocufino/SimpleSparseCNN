#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$(pwd)
echo "Starting Segmentation Training..."
python train/train.py
echo "Training Complete!"
