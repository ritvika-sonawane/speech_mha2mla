#!/bin/bash

echo "=========================================="
echo "Preparing LibriSpeech 100h Dataset"
echo "=========================================="

# Create data directory
mkdir -p data/librispeech

cd data/librispeech

# Download LibriSpeech train-clean-100 (100 hours)
echo "Downloading LibriSpeech train-clean-100..."
if [ ! -f "train-clean-100.tar.gz" ]; then
    wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
else
    echo "train-clean-100.tar.gz already exists, skipping download..."
fi

# Download test sets
echo "Downloading test-clean..."
if [ ! -f "test-clean.tar.gz" ]; then
    wget http://www.openslr.org/resources/12/test-clean.tar.gz
else
    echo "test-clean.tar.gz already exists, skipping download..."
fi

echo "Downloading test-other..."
if [ ! -f "test-other.tar.gz" ]; then
    wget http://www.openslr.org/resources/12/test-other.tar.gz
else
    echo "test-other.tar.gz already exists, skipping download..."
fi

# Extract archives
echo "Extracting archives..."
for file in train-clean-100.tar.gz test-clean.tar.gz test-other.tar.gz; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xzf "$file"
    fi
done

echo ""
echo "=========================================="
echo "Data preparation complete!"
echo "=========================================="
echo ""
echo "Dataset structure:"
echo "  - Training: data/librispeech/LibriSpeech/train-clean-100/"
echo "  - Test Clean: data/librispeech/LibriSpeech/test-clean/"
echo "  - Test Other: data/librispeech/LibriSpeech/test-other/"
echo ""
