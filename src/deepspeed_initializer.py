import torch
import json

deepspeed_config_path = "./gpt/deepspeed_config.json"

gpu_count = torch.cuda.device_count()
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 512,
        "loss_scale_window": 10000,
        "hysteresis": 10,
        "min_loss_scale": 1
        },
    "zero_optimization": {"stage": 2}
}

with open(deepspeed_config_path, "w") as f:
    json.dump(ds_config, f, indent=2)
