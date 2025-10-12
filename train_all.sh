#!/bin/bash

# List of Python scripts with their respective arguments

scripts=(
    # # # SE
    "train_SE.py -c fixed -f T2_M2 -g 1 -e 100 -l 1e-4 -b 16"
    "train_SE.py -c fixed -f T2_M10 -g 1 -e 100 -l 1e-4 -b 16"
    "train_SE.py -c fixed -f T10_M2 -g 1 -e 100 -l 1e-4 -b 16"
    "train_SE.py -c fixed -f T10_M10 -g 1 -e 100 -l 1e-4 -b 16"
    # # # DE
    "train_DE.py -c fixed -f T2_M2 -g 1 -e 100 -l 1e-4 -b 16"
    "train_DE.py -c fixed -f T2_M10 -g 1 -e 100 -l 1e-4 -b 16"
    "train_DE.py -c fixed -f T10_M2 -g 1 -e 100 -l 1e-4 -b 16"
    "train_DE.py -c fixed -f T10_M10 -g 1 -e 100 -l 1e-4 -b 16"
    # # # SE
    # "train_SE.py -c fixed -f n_T2_M2 -g 1 -e 50 -l 1e-4 -b 16"
    # "train_SE.py -c fixed -f n_T2_M10 -g 1 -e 50 -l 1e-4 -b 16"
    # "train_SE.py -c fixed -f n_T10_M2 -g 1 -e 50 -l 1e-4 -b 16"
    # "train_SE.py -c fixed -f n_T10_M10 -g 1 -e 50 -l 1e-4 -b 16"
    # # # DE
    # "train_DE.py -c fixed -f n_T2_M2 -g 1 -e 50 -l 1e-4 -b 16"
    # "train_DE.py -c fixed -f n_T2_M10 -g 1 -e 50 -l 1e-4 -b 16"
    # "train_DE.py -c fixed -f n_T10_M2 -g 1 -e 50 -l 1e-4 -b 16"
    # "train_DE.py -c fixed -f n_T10_M10 -g 1 -e 50 -l 1e-4 -b 16"

    # # # SE
    "train_SE.py -c fixed -f T2_M20 -g 0 -e 100 -l 1e-4 -b 16"
    "train_SE.py -c fixed -f T2_M100 -g 0 -e 100 -l 1e-4 -b 16"
    # # # DE
    "train_DE.py -c fixed -f T2_M20 -g 0 -e 100 -l 1e-4 -b 16"
    "train_DE.py -c fixed -f T2_M100 -g 0 -e 100 -l 1e-4 -b 16"
)

# Name of the conda environment
conda_env="torch"

# Loop through the scripts and create a screen for each
for script in "${scripts[@]}"; do
    # Extract the base name of the script (first word) to use as the screen name
    screen_name=$(basename "$(echo $script | awk '{print $1}')" .py)
    
    # Start a new detached screen and execute commands
    screen -dmS "$screen_name" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh;  # Update this path if your conda is located elsewhere
        conda activate $conda_env;
        python $script;
        exec bash
    "
    echo "Started screen '$screen_name' for script '$script'."
done
