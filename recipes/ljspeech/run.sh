#!/bin/bash

set -e

echo "===== UniMamba-TTS Training Pipeline ====="

CONFIG="configs/config.yaml"

echo ""
echo "Step 1: Downloading LJSpeech dataset..."
bash recipes/ljspeech/prepare.sh

echo ""
echo "Step 2: Preprocessing dataset..."
python unimamba_tts/cli/preprocess.py --config $CONFIG --stage 0

echo ""
echo "Step 3: Training UniMamba-TTS..."
python unimamba_tts/cli/train.py --config $CONFIG

echo ""
echo "===== Training Pipeline Completed ====="
