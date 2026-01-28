#!/bin/bash

DATASET_DIR="data/raw/ljspeech"

mkdir -p $DATASET_DIR

echo "Downloading LJSpeech dataset..."
aria2c -x 16 -s 16 -k 1M https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -d $DATASET_DIR

echo "Extracting dataset..."
tar -xjf $DATASET_DIR/LJSpeech-1.1.tar.bz2 -C $DATASET_DIR

mv $DATASET_DIR/LJSpeech-1.1/* $DATASET_DIR/
rm -rf $DATASET_DIR/LJSpeech-1.1
rm $DATASET_DIR/LJSpeech-1.1.tar.bz2

echo "LJSpeech dataset downloaded and extracted to $DATASET_DIR"
