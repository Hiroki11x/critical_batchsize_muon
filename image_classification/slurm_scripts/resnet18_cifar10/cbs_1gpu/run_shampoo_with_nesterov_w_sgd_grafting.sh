#!/bin/bash
WANDB_ENTITY="anonymous_user"
WANDB_PROJECT_NAME="v4_shampoo_cbs"

BASE_BS_PER_GPU=512
BASE_EPOCHS=100
ARCH="resnet18"

OPTIMIZER="shampoo"
TARGET_TRAIN_ACC=95
TARGET_TEST_ACC=90

SHAMPOO_LR_LIST=(0.1 0.01 0.001)
BS_PER_GPU_LIST=(4096 2048 1024 512 256 128 64 32 16 8 4 2)
WD_LAMBDA_LIST=(0)
SHAMPOO_MOMENTUM_LIST=(0.9)
SHAMPOO_GRAFT_TYPE_LIST=("sgd")
SHAMPOO_START_PRECOND_LIST=(250)

# Debug experiment run

counter=0
lower_bound=0
upper_bound=100

for LR in "${SHAMPOO_LR_LIST[@]}"; do
    for BS_PER_GPU in "${BS_PER_GPU_LIST[@]}"; do
        for WD_LAMBDA in "${WD_LAMBDA_LIST[@]}"; do
            for MOMENTUM in "${SHAMPOO_MOMENTUM_LIST[@]}"; do
                for GRAFT_TYPE in "${SHAMPOO_GRAFT_TYPE_LIST[@]}"; do
                    for START_PRECOND in "${SHAMPOO_START_PRECOND_LIST[@]}"; do

                        SCALE_FACTOR=$(bc <<<"scale=8; sqrt(${BS_PER_GPU}/${BASE_BS_PER_GPU})")
                        SCALED_LR=$(bc <<<"scale=8; ${LR} * ${SCALE_FACTOR}")
                        SCALED_START_PRECOND=$(bc <<<"scale=0; ${START_PRECOND} * ${SCALE_FACTOR}")

                        SCALED_EPOCHS=$(bc <<<"scale=0; ${BASE_EPOCHS} * ${BASE_BS_PER_GPU} / ${BS_PER_GPU}")

                        WANDB_EXPNAME="${OPTIMIZER}_eta${SCALED_LR}_bs${BS_PER_GPU}_graft${GRAFT_TYPE}"
                        ARGS="--optimizer ${OPTIMIZER} \
                            --lr ${SCALED_LR} \
                            --momentum ${MOMENTUM} \
                            --nesterov \
                            --arch ${ARCH} \
                            --target_train_accuracy ${TARGET_TRAIN_ACC} \
                            --target_test_accuracy ${TARGET_TEST_ACC} \
                            --weight_decay ${WD_LAMBDA} \
                            --bs_per_gpu ${BS_PER_GPU} \
                            --epochs ${SCALED_EPOCHS} \
                            --shampoo_epsilon 1e-8 \
                            --shampoo_update_freq 1 \
                            --shampoo_precond_freq 1 \
                            --shampoo_start_precond ${SCALED_START_PRECOND} \
                            --shampoo_block_size 8192 \
                            --shampoo_graft_type ${GRAFT_TYPE} \
                            --shampoo_graft_epsilon 1e-8 \
                            --shampoo_graft_beta1 0.9 \
                            --shampoo_graft_beta2 0.999 \
                            --wandb_project ${WANDB_PROJECT_NAME} \
                            --wandb_entity ${WANDB_ENTITY} \
                            --wandb_expname ${WANDB_EXPNAME} \
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
        done
    done
done

echo "Total experiments submitted: $counter" 