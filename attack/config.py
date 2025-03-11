import argparse
from datetime import datetime
from functools import partial
from typing import List, Tuple, Union

from pydantic import BaseModel


from zoology.utils import import_from_str

from zoology.base_config import BaseConfig, FunctionConfig, ModuleConfig

class DataSegmentConfig(BaseConfig):
    """
    This class should be subclassed to define per task. For example, MQARConfig
    """
    vocab_size: int = 8_192
    num_examples: int = 1_000
    input_seq_len: int = 64

    def build(self, **kwargs):
        raise NotImplementedError()
    
class DataConfig(BaseConfig):
    train_configs: List[DataSegmentConfig]
    test_configs: List[DataSegmentConfig]

    # can pass a tuple if you want a different batch size for train and test
    batch_size: Union[int, Tuple[int, int]] = 32

    seed: int = 123

    cache_dir: str = None
    force_cache: bool = False 

class ModelConfig(BaseConfig):
    sequence_mixer: ModuleConfig = None
    state_mixer: ModuleConfig = ModuleConfig(
        name="zoology.mixers.mlp.MLP", 
        kwargs={"hidden_mult": 4}
    )

    d_model: int = 128
    n_layers: int = 2

    # for discrete input, number of tokens
    # for continous input, number of input channels
    vocab_size: int = 8_192

    # number of output channels
    # if output_dim < 0, output_dim will be set to vocab_size
    output_dim: int = -1

    # embedders: token / linproj
    embedder_type: str = "token"
    max_position_embeddings: int = 64
    learnable_word_embeddings: bool = True

    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    drop_path: float = 0.0
    layer_norm_epsilon: float = 1e-5
    pad_vocab_size_multiple: int = 1

    block_type: str = "TransformerBlock"
    name: str = "default"

    # classifier pooling: 'max' / 'mean' / 'clstok' / empty
    # if empty, no pooling for seq2seq tasks
    classifier_pool: str = ''


class LoggerConfig(BaseConfig):
    # 'wandb' / 'text'
    logger_type: str = 'wandb'

    project_name: str = None
    entity: str = None

    # number of output samples to visualize
    # if 0, disable output vis.
    num_vis_output: int = 0
    vis_meta: dict = {}

    max_num_ckpts: int = 3


class TrainConfig(BaseConfig):
    data: DataConfig
    model: ModelConfig
    logger: LoggerConfig = LoggerConfig()

    # loss function: 'mse' / 'ce'
    loss_fn: str = 'ce'
    # evaluation metrics: 'accuracy', 'psnr'
    eval_metrics: List[str] = ['accuracy']

    max_epochs: int = 100

    # stop training once this metric reaches the threshold
    # set metric to None to disable early stopping
    early_stopping_metric: str = "valid/accuracy"
    early_stopping_threshold: float = 0.99
    slice_keys: List[str] = []

    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 123

    launch_id: str = None
    sweep_id: str = None
    run_id: str = "default"

    output_dir: str = ''
    eval_only: bool = False
    eval_name: str = ''
