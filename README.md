# FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection
![FA framework](framework.png)
The paper has been accepted by ICCV2025  
Arxiv: https://arxiv.org/abs/2507.04511

## Requirements
#### Installation
Create a conda environment and install dependencies (you do not need to install the toolbox Dassl.pytorch first):
```bash
pip install -r requirements.txt
```

## Get Started
### Configs
The running configurations can be modified in `configs/my_config.yaml`.  
The ID dataset configurations can be modified in `my_dataset/`.

### Train
`CUDA_VISIBLE_DEVICES=3 python main.py --config configs/my_config.yaml --is_train 1`

### Inference
`CUDA_VISIBLE_DEVICES=2 python main.py --config configs/my_config.yaml --is_train 0`  
The specific paths for different OOD datasets can be modified in the inference section of the `main.py` code.


