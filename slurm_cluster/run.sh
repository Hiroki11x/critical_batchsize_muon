#!/bin/bash
#SBATCH --partition=long                # Ask for unkillable job
#SBATCH --cpus-per-task=4                    # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                     # Ask for 1 GPU
#SBATCH --mem=16G                             # Ask for 10 GB of RAM
#SBATCH --time=01:00:00                        # The job will run for 3 hours
#SBATCH -o slurm-%j.out  # Write the log on tmp1


# ======== Module, Virtualenv and Other Dependencies ======
source ./env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH


# ======== Execution ========
PROGRAM_NAME="main.py"
ARGS=$@
pushd ../src

EXEC_CMD="python ${PROGRAM_NAME} ${ARGS}"
echo "[Command] : ${EXEC_CMD}"
eval $EXEC_CMD

popd