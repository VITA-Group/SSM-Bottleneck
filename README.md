# Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The official implementation of ICLR 2025 paper [Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing](https://arxiv.org/abs/2501.00658).

Peihao Wang, Ruisi Cai, Yuehao Wang, Jiajun Zhu, Pragya Srivastava, Zhangyang Wang, Pan Li

Our implementation and experiments are based on [Zoology](https://github.com/HazyResearch/zoology) codebase. 


## Getting Started

This repository requires the following packages:

```
# basic
numpy
einops
tqdm
click
pydantic>=2.0.0,<2.5.0
wandb
imageio

# torch
torch
torchvision

# huggingface
transformers

# for mamba
mamba_ssm
causal_conv1d
```

Note that `mamba_ssm`, `causal_conv1d` and `transformers` are frequently updated libraries. At the development of this project, we run our experiments with `mamba_ssm==1.1.4,causal_conv1d==1.1.0,transformers=4.43.3`. We have so far made the code compatible with `mamba_ssm==2.2.4`.

## Recency Bias Attack

The script files for our adversarial attack and target attack experiments are `attack/adv_attack.py` and `attack/tgt_attack.py`, respectively. You will need to modify the values of some variables to configure the script. For example, the `data_path` variable specifies the dataset path, which will be passed to the `root` parameter in [torchvision.datasets.CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html).


### Training

If you want to reproduce our results of adversarial attack and target attack experiments, you can train Mamba, Transformer, H3, and RWKV models from scratch, using the `adv_attack/adv_attack.py` script. **You will need to set `is_eval_only = False` in `adv_attack/adv_attack.py` to turn on training mode.** To set up wandb logging, you will need to change `entity_name` to your [wandb](https://wandb.ai/site/) entity name. Alternatively, you can disable wandb by setting `logger_type` to `'text'` in `LoggerConfig`. After these configurations, type the following command:

```bash
python -m attack.launch attack/adv_attack.py --name imgcls-sweep-train -p --gpus 0,1,2,3
```

**Note: The `-p` flag enables parallel running across multiple GPUs if available.**

For more command-line options, please refer to the [zoology readme](https://github.com/HazyResearch/zoology/blob/main/README.md) page.

In addition to training from scratch, you may also download our pretrained models [here](https://huggingface.co/peihaowang/ssm-bottleneck-imgcls-attack) and put them under `logs/imgcls`.


### Evaluation

After training (or downloading the pretrained checkpoints), you can reset `is_eval_only = True` and run the following commands to perform evaluation under recency bias attack:

```bash
# Adversarial attack
python -m attack.launch attack/adv_attack.py --name imgcls-sweep-eval -p --gpus 0,1,2,3

# Target attack
python -m attack.launch attack/tgt_attack.py --name imgcls-sweep-eval -p --gpus 0,1,2,3
```

Evaluation results for each model will be saved in the corresponding `logs/imgcls/{model_name}` directory. Specifically, adversarial attack results will be saved in the `eval_0099_advatk.txt` file. Target attack results will be saved in `eval_0099_atklabel{attack_label}.txt` files.

## Multi-Query Associative Recall

We provide the script `associative_recall/multi_depth_mqra.py` to compare different models across varying depths on the **Multi-Query Associative Recall (MQAR)** task. This script evaluates attention, Mamba, H3, and Polarized Mamba (with zero-polarized, one-polarized, or both channels).

```bash
python -m attack.launch associative_recall/multi_depth_mqra.py --name multi-depth-mqra -p --gpus 0,1,2,3
```

**Note: The `-p` flag enables parallel training across multiple GPUs if available.**

By default, the vocabulary size is 8192, and the training data consists of sequences of varying lengths. Models are tested with 64, 128, and 256 key-value pairs. To customize training and evaluation, please modify the configurations in `associative_recall/multi_depth_mqra.py`.

Evaluation results for each model will be saved under the `logs/ssm_mqar` directory.

## Citation

If you find this work or our code implementation helpful for your own resarch or work, please cite our paper.
```
@inproceedings{wang2024understanding,
title={Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing},
author={Wang, Peihao and Cai, Ruisi and Wang, Yuehao and Zhu, Jiajun and Srivastava, Pragya and Wang, Zhangyang and Li, Pan},
booktitle={International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=pymXpl4qvi},
}
```