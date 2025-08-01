#!/bin/bash
export WANDB_ENTITY="anonymous_user"
WANDB_PROJECT_NAME="v4_muon_convergence"
OPTIMIZER="muon"
BS_PER_GPU=512
EPOCHS=100
LAMBDA=0.125 # 1/lambda = 8
ADAM_LR_LIST=(0.01 0.001)
ARCH="resnet18"
TARGET_TRAIN_ACC=85
TARGET_TEST_ACC=80


for LR in 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0; do
    for ADAM_LR in "${ADAM_LR_LIST[@]}"; do 
        WANDB_EXPNAME="muon_eta${LR}_lt_inv_lambda"
        ARGS="--optimizer ${OPTIMIZER} \
            --nesterov \
            --lr ${LR} \
            --adam_lr ${ADAM_LR} \
            --arch ${ARCH} \
            --target_train_accuracy ${TARGET_TRAIN_ACC} \
            --target_test_accuracy ${TARGET_TEST_ACC} \
            --weight_decay ${LAMBDA} \
            --bs_per_gpu ${BS_PER_GPU} \
            --epochs ${EPOCHS} \
            --wandb_project ${WANDB_PROJECT_NAME} \
            --wandb_entity ${WANDB_ENTITY} \
            --wandb_expname ${WANDB_EXPNAME}"
        CMD="sbatch run.sh $ARGS"
        pushd ../
        echo "Running command: $CMD"
        eval $CMD
        popd
    done
done

for LR in 12.0 16.0 24.0 32.0; do
    for ADAM_LR in "${ADAM_LR_LIST[@]}"; do 
        WANDB_EXPNAME="muon_eta${LR}_gt_inv_lambda"
        ARGS="--optimizer ${OPTIMIZER} \
            --nesterov \
            --lr ${LR} \
            --adam_lr ${ADAM_LR} \
            --arch ${ARCH} \
            --target_train_accuracy ${TARGET_TRAIN_ACC} \
            --target_test_accuracy ${TARGET_TEST_ACC} \
            --weight_decay ${LAMBDA} \
            --bs_per_gpu ${BS_PER_GPU} \
            --epochs ${EPOCHS} \
            --wandb_project ${WANDB_PROJECT_NAME} \
            --wandb_entity ${WANDB_ENTITY} \
            --wandb_expname ${WANDB_EXPNAME}"
        CMD="sbatch run.sh $ARGS"
        pushd ../
        echo "Running command: $CMD"
        eval $CMD
        popd
    done
done