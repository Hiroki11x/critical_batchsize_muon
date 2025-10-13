#!/bin/bash
WANDB_ENTITY="anonymous_user"
WANDB_PROJECT_NAME="v4_msgd_cbs"

BASE_BS_PER_GPU=512
BASE_EPOCHS=100

ARCH="resnet18"

OPTIMIZER="msgd"
TARGET_TRAIN_ACC=95
TARGET_TEST_ACC=90

MSGD_LR_LIST=(0.001 0.0005 0.0001)
BS_PER_GPU_LIST=(4096 2048 1024 512 256 128 64 32 16 8 4 2)
WD_LAMBDA_LIST=(0.1 0.01 0.001 0.0001 0)

counter=0
for BS_PER_GPU in "${BS_PER_GPU_LIST[@]}"; do
    for WD_LAMBDA in "${WD_LAMBDA_LIST[@]}"; do
        
        SCALE_FACTOR=$(bc <<< "scale=8; ${BS_PER_GPU}/${BASE_BS_PER_GPU}")

        SCALED_EPOCHS=$(bc <<< "scale=0; ${BASE_EPOCHS} * ${BASE_BS_PER_GPU} / ${BS_PER_GPU}")

        for MSGD_LR in "${MSGD_LR_LIST[@]}"; do 

            SCALED_LR=$(bc <<< "scale=8; ${MSGD_LR} * ${SCALE_FACTOR}")
            WANDB_EXPNAME="${OPTIMIZER}_eta${SCALED_LR}_bs${BS_PER_GPU}"
            ARGS="--optimizer ${OPTIMIZER} \
                --lr ${SCALED_LR} \
                --arch ${ARCH} \
                --target_train_accuracy ${TARGET_TRAIN_ACC} \
                --target_test_accuracy ${TARGET_TEST_ACC} \
                --weight_decay ${WD_LAMBDA} \
                --bs_per_gpu ${BS_PER_GPU} \
                --epochs ${SCALED_EPOCHS} \
                --wandb_project ${WANDB_PROJECT_NAME} \
                --wandb_entity ${WANDB_ENTITY} \
                --wandb_expname ${WANDB_EXPNAME} \
                --wandb_offline"

            RUN_SCRIPT="run.sh"

            CMD="sbatch ${RUN_SCRIPT} $ARGS"
            pushd ../
            echo "Running command: $CMD"
            eval $CMD
            popd

            counter=$((counter + 1))
        done
    done
done

echo "Total experiments submitted: $counter"
