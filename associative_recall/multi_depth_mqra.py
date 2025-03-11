from zoology.base_config import FunctionConfig, ModuleConfig
from associative_recall.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from associative_recall.data.associative_recall import MQARConfig, LongRangeMQARConfig

import os
import numpy as np

## data configs
VOCAB_SIZE = 8_192
SEQ_LEN = 1024
DATA_CONFIG = {
    "train_power_a": 0.0,
    "test_power_a": 0.0,
    "random_non_queries": bool(int(os.environ.get('RAND_PLACEHOLDER', False)))
}


## training hyper-parameters
NUM_EPOCHS = 64
BATCH_SIZE = 128
LR = 1e-3


## model specifications
# hidden dimension
D_MODEL = 128
# depth choices
LS_MODEL = [2, 4]

ALL_MIXERS = {

    "transformer": {
        "block_type": "TransformerBlock",
        "mixer_config": ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        ),
        "state_config": ModuleConfig(
            name="zoology.mixers.mlp.MLP", 
            kwargs={"hidden_mult": 4}
        ),
        "max_pos_embed": SEQ_LEN,
    },

    "mamba": {
        "block_type": "MambaBlock",
        "mixer_config": ModuleConfig(
            name="zoology.mixers.mamba.Mamba",
            kwargs={"d_state": 16}
        ),
        "state_config": dict(name="torch.nn.Identity", kwargs={}),
        "max_pos_embed": 0, # Disable positional embeddings
    },


    "polar_mamba": {
        "block_type": "PolarizedMambaBlock",
        "mixer_config": ModuleConfig(
            name="zoology.mixers.polar_mamba.PolarizedMamba",
            kwargs={"d_state": 16, "d_zeros": 1, "d_ones": 1}
        ),
        "state_config": dict(name="torch.nn.Identity", kwargs={}),
        "max_pos_embed": 0, # Disable positional embeddings
    },


    "polar_mamba_zero": {
        "block_type": "PolarizedMambaBlock",
        "mixer_config": ModuleConfig(
            name="zoology.mixers.polar_mamba.PolarizedMamba",
            kwargs={"d_state": 16, "d_zeros": 1, "d_ones": 0}
        ),
        "state_config": dict(name="torch.nn.Identity", kwargs={}),
        "max_pos_embed": 0, # Disable positional embeddings
    },


    "polar_mamba_one": {
        "block_type": "PolarizedMambaBlock",
        "mixer_config": ModuleConfig(
            name="zoology.mixers.polar_mamba.PolarizedMamba",
            kwargs={"d_state": 16, "d_zeros": 0, "d_ones": 1}
        ),
        "state_config": dict(name="torch.nn.Identity", kwargs={}),
        "max_pos_embed": 0, # Disable positional embeddings
    },

    "h3": {
        "block_type": "TransformerBlock",
        "mixer_config": ModuleConfig(
            name="zoology.mixers.h3.h3.H3",
            kwargs={
                "l_max": 256,
                "d_state": 512 / 4,
                "head_dim": 2
            }
        ),
        "state_config": ModuleConfig(
            name="zoology.mixers.mlp.MLP", 
            kwargs={"hidden_mult": 4}
        ),
        "max_pos_embed": 0, # Disable positional embeddings
    },
}

configs = []

for MODEL_NAME, MODEL_CONFIG in ALL_MIXERS.items():

    for L_MODEL in LS_MODEL:

        config = TrainConfig(

            do_train = True,

            data=DataConfig(
                cache_dir="./data_cache/mqar",

                train_configs = [    
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4, **DATA_CONFIG),
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8, **DATA_CONFIG),
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16, **DATA_CONFIG),
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32, **DATA_CONFIG),
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64, **DATA_CONFIG),
                ] + (
                    [
                        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=20_000, num_kv_pairs=32, **DATA_CONFIG),
                        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=20_000, num_kv_pairs=64, **DATA_CONFIG),
                        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=20_000, num_kv_pairs=128, **DATA_CONFIG),
                    ] if SEQ_LEN >= 512 else []
                ) + (
                    [
                        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=20_000, num_kv_pairs=64, **DATA_CONFIG),
                        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=20_000, num_kv_pairs=128, **DATA_CONFIG),
                        MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=20_000, num_kv_pairs=256, **DATA_CONFIG),
                    ] if SEQ_LEN >= 1024 else []
                ),
        
                test_configs = [
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=SEQ_LEN, num_examples=1_000, num_kv_pairs=SEQ_LEN//16, **DATA_CONFIG),
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=SEQ_LEN, num_examples=1_000, num_kv_pairs=SEQ_LEN//8, **DATA_CONFIG),
                    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=SEQ_LEN, num_examples=1_000, num_kv_pairs=SEQ_LEN//4, **DATA_CONFIG)
                ],

                batch_size = (BATCH_SIZE, BATCH_SIZE // 8),
            ),


            model=ModelConfig(
                vocab_size=VOCAB_SIZE,
                max_position_embeddings=MODEL_CONFIG['max_pos_embed'], 
                d_model=D_MODEL,
                n_layers=L_MODEL,
                block_type = MODEL_CONFIG['block_type'],
                sequence_mixer=MODEL_CONFIG['mixer_config'],
                state_mixer=MODEL_CONFIG['state_config'],
                name=MODEL_NAME,
            ),

            max_epochs=NUM_EPOCHS,
            learning_rate=LR,
            slice_keys=["num_kv_pairs"],

            run_id = f"{MODEL_NAME}-seqlen{SEQ_LEN}-layer{L_MODEL}-dmodel{D_MODEL}",
            logger=LoggerConfig(
                root_path="./logs",
                project_name="ssm_mqar",
                entity=""
            )
        )

        configs.append(config)
