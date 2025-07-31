import torch
import json

deepspeed_config_path = "./gpt/deepspeed_config.json"

gpu_count = torch.cuda.device_count()
ds_config = {
    "train_batch_size": 72,
    "gradient_accumulation_steps": 3,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5
        }
    },
    "lr_scheduler":{
        "type": "OneCycle",
        "params":{
            "total_steps": 125000,
            "max_lr": 3e-5,
            "pct_start": 0.3,
            "anneal_strategy": "linear",
            "cycle_momentum": False,
        }
    },

    "fp16": {
        "enabled": False
        },
    "zero_optimization": {"stage": 1},
    "gradient_clipping": 1.0
}

with open(deepspeed_config_path, "w") as f:
    json.dump(ds_config, f, indent=2)
