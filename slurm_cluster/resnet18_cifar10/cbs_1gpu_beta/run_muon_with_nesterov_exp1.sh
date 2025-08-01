#!/bin/bash
WANDB_ENTITY="anonymous_user"
WANDB_PROJECT_NAME="v4_muon_cbs_beta"

BASE_BS_PER_GPU=512
BASE_EPOCHS=100

ARCH="resnet18"

OPTIMIZER="muon"
TARGET_TRAIN_ACC=95
TARGET_TEST_ACC=90

ADAM_LR_LIST=(0.01 0.001 0.0001)
MUON_LR_LIST=(0.001 0.005 0.01)
BS_PER_GPU_LIST=(4096 2048 1024 512 256 128 64 32 16 8 4 2)
WD_LAMBDA_LIST=(0)
MUON_BETA_LIST=(0.7 0.8 0.9 0.95 0.99 0.999)

# 648 EXP RUN

counter=0
lower_bound=0
upper_bound=200

for LR in "${MUON_LR_LIST[@]}"; do
    for BS_PER_GPU in "${BS_PER_GPU_LIST[@]}"; do
        for WD_LAMBDA in "${WD_LAMBDA_LIST[@]}"; do

            for MUON_BETA in "${MUON_BETA_LIST[@]}"; do

                MOMENTUM=${MUON_BETA}

                SCALE_FACTOR=$(bc <<<"scale=8; sqrt(${BS_PER_GPU}/${BASE_BS_PER_GPU})")
                SCALED_LR=$(bc <<<"scale=8; ${LR} * ${SCALE_FACTOR}")

                SCALED_EPOCHS=$(bc <<<"scale=0; ${BASE_EPOCHS} * ${BASE_BS_PER_GPU} / ${BS_PER_GPU}")

                for ADAM_LR in "${ADAM_LR_LIST[@]}"; do

                    SCALED_ADAM_LR=$(bc <<<"scale=8; ${ADAM_LR} * ${SCALE_FACTOR}")
                    WANDB_EXPNAME="${OPTIMIZER}_eta${SCALED_LR}_bs${BS_PER_GPU}"
                    ARGS="--optimizer ${OPTIMIZER} \
                        --nesterov \
                        --lr ${SCALED_LR} \
                        --adam_lr ${SCALED_ADAM_LR} \
                        --momentum ${MOMENTUM} \
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

echo "Total experiments submitted: $counter"
