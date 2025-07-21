#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0 
export MAX_JOBS=4

MODEL_CONFIG_VERSION=1 deepspeed --num_gpus=2 ./gpt/src/model_trainer.py