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
    max_position_embeddings: int = 64
    learnable_word_embeddings: bool = True
    vocab_size: int = 8_192

    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    drop_path: float = 0.0
    layer_norm_epsilon: float = 1e-5
    pad_vocab_size_multiple: int = 1

    block_type: str = "TransformerBlock"
    name: str = "default"

class LoggerConfig(BaseConfig):

    root_path: str = './'

    project_name: str = None
    entity: str = None
    

class TrainConfig(BaseConfig):
    data: DataConfig
    model: ModelConfig
    logger: LoggerConfig = LoggerConfig()

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

    do_train: bool = True
    do_eval: bool = True
