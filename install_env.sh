#!/bin/bash
# 
# Installer for package
# 
# Run: bash install_env.sh
# 

echo 'Creating package environment:'

# Create conda env
conda env create -f environment.yml
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh

conda activate MEDL
echo 'Created and activated environment:' $(which python)
pip install -e .

# Check torch works as expected
echo 'Checking torch version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'
