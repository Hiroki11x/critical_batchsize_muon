#!/bin/bash
WANDB_ENTITY="anonymous_user"
WANDB_PROJECT_NAME="v1_lr_search_adamw_cbs_vgg16_cifar100"
# WANDB_PROJECT_NAME="debug_v1_lr_search_adamw_cbs_vgg16_cifar100"

BASE_BS_PER_GPU=512
BASE_EPOCHS=100

ARCH="vgg16_bn"
DATASET="cifar100"

OPTIMIZER="adamw"
# https://github.com/weiaicunzai/pytorch-cifar100/tree/master
TARGET_TRAIN_ACC=80  # Lower target for CIFAR-100
TARGET_TEST_ACC=60   # Lower target for CIFAR-100

ADAM_LR_LIST=(1.0 0.1 0.01 0.001 0.0001 0.00001)
# ADAM_LR_LIST=(0.1)
# BS_PER_GPU_LIST=(2048 512 128 32 8 2)
BS_PER_GPU_LIST=(512)
WD_LAMBDA_LIST=(0)
# WD_LAMBDA_LIST=(0.1 0.01 0.001 0.0001)


counter=0
lower_bound=0
upper_bound=1000

for LR in "${ADAM_LR_LIST[@]}"; do
    for BS_PER_GPU in "${BS_PER_GPU_LIST[@]}"; do
        for WD_LAMBDA in "${WD_LAMBDA_LIST[@]}"; do

            SCALE_FACTOR=$(bc <<<"scale=8; sqrt(${BS_PER_GPU}/${BASE_BS_PER_GPU})")
            SCALED_LR=$(bc <<<"scale=8; ${LR} * ${SCALE_FACTOR}")

            SCALED_EPOCHS=$(bc <<<"scale=0; ${BASE_EPOCHS} * ${BASE_BS_PER_GPU} / ${BS_PER_GPU}")

            WANDB_EXPNAME="${OPTIMIZER}_eta${SCALED_LR}_bs${BS_PER_GPU}"
            ARGS="--optimizer ${OPTIMIZER} \
                --lr ${SCALED_LR} \
                --arch ${ARCH} \
                --dataset ${DATASET} \
                --target_train_accuracy ${TARGET_TRAIN_ACC} \
                --target_test_accuracy ${TARGET_TEST_ACC} \
                --weight_decay ${WD_LAMBDA} \
                --bs_per_gpu ${BS_PER_GPU} \
                --epochs ${SCALED_EPOCHS} \
                --early_stopping \
                --wandb_project ${WANDB_PROJECT_NAME} \
                --wandb_entity ${WANDB_ENTITY} \
                --wandb_expname ${WANDB_EXPNAME}\
                --wandb_offline"

            RUN_SCRIPT="run.sh"

            CMD="sbatch ${RUN_SCRIPT} $ARGS"
            pushd ../../

            if [ ${counter} -ge ${lower_bound} ] && [ ${counter} -lt ${upper_bound} ] ; then
                echo "Exp-${counter}: ${CMD}"
                eval $CMD
            else
                echo "Skip Exp-${counter}"
            fi

            popd

            counter=$((counter + 1))
        done
    done
done

echo "Total experiments submitted: $counter" 