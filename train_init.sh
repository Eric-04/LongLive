# #!/bin/bash

# Project path and config
CONFIG=configs/longlive_train_init.yaml
LOGDIR=logs
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=8 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --no-one-logger \
  --disable-wandb
  # --wandb-save-dir $WANDB_SAVE_DIR
