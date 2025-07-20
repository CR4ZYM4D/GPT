import torch
import json

deepspeed_config_path = "./gpt/deepspeed_config.json"

gpu_count = torch.cuda.device_count()
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
    "tensor_parallel": {"tp_size": gpu_count}
}

with open(deepspeed_config_path, "w") as f:
    json.dump(ds_config, f, indent=2)
