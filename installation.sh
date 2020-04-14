#!/bin/bash

# 0. Check python and pip3
(python3 -V && pip3 -V) &>/dev/null
if [[ $? -ne 0 ]]; then
    echo "Error: please check whether 'python3' and 'pip3' is valid"
    exit 1
fi

# 1. Create virtual environment
conda -V &>/dev/null
is_conda=$?
if [[ $is_conda -eq 0 ]]; then
    echo "1. Creating conda environment: image_quality"
    conda create -y --name image_quality python=3.6
    source activate image_quality
else
    echo "1. Creating virtual environment: .env"
    python3 -m venv .env
    source .env/bin/activate
fi

# 2. Install tensorflow backend
nvidia-smi &>/dev/null
is_nvidia=$?
if [[ $is_nvidia -eq 0 ]]; then
    echo "2. Installing backend: tensorflow-gpu==1.12.0"
    pip install numpy==1.15.4 Keras==2.2.4 tensorflow-gpu==1.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "2. Installing backend: tensorflow==1.12.0"
    pip install numpy==1.15.4 Keras==2.2.4 tensorflow==1.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 3. Install dependencies
echo "3. Installing the rest dependencies"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. Deactivate
if [[ $is_conda -eq 0 ]]; then
    source deactivate
else
    deactivate
fi
