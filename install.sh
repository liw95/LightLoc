conda deactivate

# Set environment variables
export ENV_NAME=LightLoc
export PYTHON_VERSION=3.8
export PYTORCH_VERSION=1.11.0
export CUDA_VERSION=11.3

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

# Install PyTorch
conda install pytorch==$PYTORCH_VERSION torchvision torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=$CUDA_VERSION -c pytorch -y
# Install MinkowskiEngine
conda install openblas-devel -c anaconda -y
pip install pip==22.2.1
# Downgrade the version of setuptools
pip uninstall setuptools -y
pip install setuptools==69.5.1
# gcc g++ 9.5
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Install pip packages
pip install matplotlib
pip install tqdm
pip install h5py
pip install pandas
pip install transforms3d
pip install open3d
