IN BASH
#!/bin/bash

# Activate your virtual environment
source /data/grte4343/thesis_groundtruth/zerosumgame/venv/bin/activate

# Create scratch temp folder if it doesn't exist
mkdir -p /scratch/$USER/tmp

# Export TMPDIR so pip uses scratch for temp/cache files
export TMPDIR=/scratch/$USER/tmp

# Upgrade pip and tools (optional but recommended)
pip install --upgrade pip setuptools wheel

# Install packages (add or remove as needed)
pip install convokit pandas --quiet

echo "Setup complete!"

