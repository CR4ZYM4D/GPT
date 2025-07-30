import torch
import json

deepspeed_config_path = "./gpt/deepspeed_config.json"

gpu_count = torch.cuda.device_count()
ds_config = {
    "train_batch_size": 512,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5
        }
    },
    "bf16":{
        "enabled": True
    },
    "zero_optimization": {"stage": 2},
    "gradient_clipping": 1.0
}

with open(deepspeed_config_path, "w") as f:
    json.dump(ds_config, f, indent=2)
