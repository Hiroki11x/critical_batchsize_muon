module purge
module load your_module
echo "module list"
module list

PYTHON_PATH="/home/user/env/py310_lm/bin"
export PYTHON_PATH
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
export RUSTFLAGS="-A invalid_reference_casting"

# Name of Cluster
CLUSTER_NAME="mila_cluster"
export CLUSTER_NAME

# Dataset Dir
DATA_DIR_PATH="/home/user/.cache/huggingface/datasets"
export DATA_DIR_PATH

# Experiment Root
export EXP_ROOT="/home/user/workspace/"

# SSL
export PYTHONHTTPSVERIFY=0
export SSL_CERT_DIR=/etc/ssl/certs
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# NCCL
# export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALLREDUCE_TIMEOUT_MS=7200000
export NCCL_TIMEOUT=7200 #120 min