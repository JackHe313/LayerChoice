#!/bin/bash

# Directory containing subdirectories to process
samples_dir="ddpm_images"

# Directory to store the FID scores
output_dir="fid_scores_time"
mkdir -p "$output_dir"

# Iterate over each directory in the samples directory
for dirname in "$samples_dir"/*; do
    if [ -d "$dirname" ]; then
        # Extract the base name of the directory
        base_dirname=$(basename "$dirname")
        merged_output_dir="${output_dir}/${base_dirname}_merge"
        
        # Perform mergeImg.py operation
        if [ -d "$merged_output_dir" ]; then
            echo "Merge directory $merged_output_dir already exists. Skipping merge..."
        else
            mkdir -p "$merged_output_dir"  # Ensure the directory exists
            python mergeImg.py "$dirname/cifar10_32_500" "$merged_output_dir"
        fi
            
        # Run pytorch_fid to calculate the FID score
        fid_score=$(python -m pytorch_fid "$merged_output_dir" "/home/jackhe/LayerChoice/processed_cifar10/fid.npz" --device cuda:5)
        
        # Extract FID score from the output
        # Assuming the FID score is printed in a way that it can be directly extracted,
        # otherwise adjust the parsing according to the actual output format of pytorch_fid
        echo "$dirname: $fid_score" >> "$output_dir/fid_score.txt"
        
        echo "Processed $dirname: $fid_score FID score appended to fid_score.txt"
    fi
done
