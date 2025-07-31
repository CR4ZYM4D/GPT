#!/bin/bash

apt update

apt upgrade -y

apt install unzip tmux nano -y

nvidia-smi

good=$(python3 -c "import torch; print(torch.cuda.is_available())")

pip3 install tensorboard transformers torchmetrics torchtext tqdm deepspeed accelerate

if [[ "$good" == "True" ]]; then

    echo "Pytorch installation is good to go"

    echo "torch version: $(python3 -c 'import torch; print(torch.__version__)')"

    exit 0

else 

    pip3 uninstall torch -y
 
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

    echo "re-installed torch" 

    exit 0

fi