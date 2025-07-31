#!/bin/bash

export NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0 
export MAX_JOBS=4

MODEL_CONFIG_VERSION=0 deepspeed --num_gpus=$NUM_GPUS ./gpt/src/model_trainer.py