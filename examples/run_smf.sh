#!/bin/bash

# Base directory for data folders
data_base="data/illustris/illustris-5-8-log1p"

# Output base directory
output_base="results_200_10"

# Parameters
num_latents=10
num_inducing=200
max_iters=2000

# Iterate over all folders and run the script
for folder in "$data_base"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        output_folder="$output_base/$folder_name"
        
        echo "Processing: $folder"
        python examples/run_smf.py \
            --data_folder "$folder" \
            --output_folder "$output_folder" \
            --num_latents $num_latents \
            --num_inducing $num_inducing \
            --max_iters $max_iters
    fi
done
