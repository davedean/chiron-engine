#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Please install it first."
    echo "You can install it using: pip install uv"
    exit 1
fi

# Create a virtual environment using uv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies using uv
uv pip install -r requirements.txt

# Run the Python script
python download_qwen.py

# Deactivate the virtual environment
deactivate