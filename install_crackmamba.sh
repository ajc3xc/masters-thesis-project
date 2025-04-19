#!/bin/bash
set -eo pipefail
# Configurable variables
ENV_PATH="/mnt/stor/ceph/gchen-lab/data/Adam/crack_mamba_env"
ENV_YML="crackmamba_env.yml"
REPO_DIR="$(pwd)/CrackMamba"
MAMBA_DIR="$REPO_DIR/mamba-1.2.0"

# Step 1: Create crackmamba_env.yml
# Making it exact just to test
cat > $ENV_YML <<EOF
name: crackmamba_env
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8.15
  - pytorch-cuda=11.6
  - pytorch=1.12
  - pip
  - pip:
      - causal_conv1d==1.1.1
      - packaging
EOF

# Step 2: Create the Conda environment
echo "Creating environment at $ENV_PATH..."
#mamba env create --prefix "$ENV_PATH" -f "$ENV_YML"

# Step 3: Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

#pip install packaging

# Step 4: Install Mamba from source
echo "Installing Mamba from source in $MAMBA_DIR..."
cd "$MAMBA_DIR"
python setup.py install

# Step 5 (Optional): Initialize submodules
read -p "Do you want to initialize Git submodules? (y/n): " INIT_SUBMODULES
if [[ "$INIT_SUBMODULES" == "y" || "$INIT_SUBMODULES" == "Y" ]]; then
  echo "Initializing submodules..."
  cd "$REPO_DIR"
  git submodule update --init --recursive
fi

echo "✅ CrackMamba setup complete!"
