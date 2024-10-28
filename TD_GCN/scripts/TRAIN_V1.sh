#!/bin/bash
RECORD=2101
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/uav-cross-subjectv1/train.yaml
#CONFIG=./config/uav-cross-subjectv2/train.yaml

START_EPOCH=50
EPOCH_NUM=60
BATCH_SIZE=2
WARM_UP=5
SEED=777

python main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 1 2 3 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED

python main.py --config ./config/uav-cross-subjectv1/train.yaml --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size 2 --test-batch-size 2 --warm_up_epoch 5 --only_train_epoch 50 --seed 777

python main.py --config ./config/uav-cross-subjectv1/train.yaml
