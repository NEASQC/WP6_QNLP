#!/bin/bash

scripts_dir="slurm_scripts"
archive_dir="slurm_scripts/scripts_archive"

# Iterate over each file in the scripts directory
for file in "$scripts_dir"/*; do
    if [ -f "$file" ]; then  # Only does the following for files, ignores subdirectories
        
	# Submit the SLURM scripts to the compute nodes
        sbatch "$file"
        
        # Move the file to the archive directory
        mv "$file" "$archive_dir"
    fi
done
