import uuid
from zoology.base_config import ModuleConfig
from adv_attack.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from adv_attack.data.img_classification import ImageClassificationConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "adv_atk-sweep-" + sweep_id


### [MODIFY HERE]

# cache directory
cache_dir = "./data_cache/cifar10_advatk"

# wandb entity name (for training)
entity_name = ""

# data path
data_path = './datasets/'

# Output path
output_path = './logs/imgcls'

# If True, only run evaluation
is_eval_only = True

####



IMG_SIZE = 32
NUM_CLASSES = 10

input_seq_len = IMG_SIZE * IMG_SIZE

configs = []

batch_size = 64
n_layers = 3
d_model = 32
d_state = 64


# If True, embed 0-255 gray-scale pixel intensity to features.
# Otherwise, linearly project 0-1 pixel RGB values to features.
is_token = False
if is_token:
    embedder_type = 'token'
    vocab_size = 256
else:
    embedder_type = 'linproj'
    vocab_size = 3


dataset_kwargs = {
    "dataset": "cifar10",
    "data_path": data_path,
    "is_token": is_token
}

train_configs = [ImageClassificationConfig(num_examples=50_000, vocab_size=vocab_size, input_seq_len=input_seq_len, is_trainset=True, **dataset_kwargs)]
test_configs = [ImageClassificationConfig(num_examples=10_000, vocab_size=vocab_size, input_seq_len=input_seq_len, is_trainset=False, **dataset_kwargs)]

if is_eval_only:
    # The lengths of corrupted region
    perturb_pix = [32, 96]
    test_configs += [ImageClassificationConfig(
            num_examples=10_000, vocab_size=vocab_size, input_seq_len=input_seq_len, is_trainset=False,
            perturb_pixels=[0, k],
            perturb_mode='randn', **dataset_kwargs) for k in perturb_pix]
    test_configs += [ImageClassificationConfig(
            num_examples=10_000, vocab_size=vocab_size, input_seq_len=input_seq_len, is_trainset=False,
            perturb_pixels=[input_seq_len - k, input_seq_len],
            perturb_mode='randn', **dataset_kwargs) for k in perturb_pix]    

model_factory_kwargs = {
    "vocab_size": vocab_size,
    "output_dim": NUM_CLASSES,
    "embedder_type": embedder_type,
}

# Pooling scheme for classifer.
# 'clstok': using a class token, like ViT.
# 'mean': average pooling over all tokens.
classifier_pool_opts = ['clstok']

models = []

# mamba 
block_type = "MambaBlock"
for classifier_pool in classifier_pool_opts:
    mixer = dict(
        name="zoology.mixers.mamba.Mamba",
        kwargs={"d_state": d_state}
    )
    model = ModelConfig(
        block_type="MambaBlock",
        d_model=d_model,
        n_layers=n_layers,
        sequence_mixer=mixer,
        state_mixer=dict(name="torch.nn.Identity", kwargs={}),
        max_position_embeddings=0,
        name="mamba",
        classifier_pool=classifier_pool,
        **model_factory_kwargs
    )
    models.append(model)


# H3
block_type = "TransformerBlock"
for classifier_pool in classifier_pool_opts:
    mixer = dict(
        name="zoology.mixers.h3.H3",
        kwargs={
            "l_max": input_seq_len + (1 if classifier_pool == 'clstok' else 0),
            "d_state": d_state,
            "head_dim": 2
        }
    )
    model = ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=n_layers,
        sequence_mixer=mixer,
        state_mixer=dict(name="torch.nn.Identity", kwargs={}),
        max_position_embeddings=0,
        name="h3",
        classifier_pool=classifier_pool,
        **model_factory_kwargs
    )
    models.append(model)


# rwkv
block_type = "TransformerBlock"
for classifier_pool in classifier_pool_opts:
    mixer = dict(
        name="zoology.mixers.rwkv.RWKVTimeMixer",
        kwargs={
            "l_max": input_seq_len + (1 if classifier_pool == 'clstok' else 0),
        },
    )
    model = ModelConfig(
        block_type="TransformerBlock",
        d_model=64,  # should be larger than other models to ensure comparable model size
        n_layers=n_layers,
        sequence_mixer=mixer,
        state_mixer=dict(name="torch.nn.Identity", kwargs={}),
        max_position_embeddings=0,
        name=f"rwkv",
        classifier_pool=classifier_pool,
        **model_factory_kwargs
    )
    models.append(model)


# attention
max_position_embeddings = input_seq_len + (1 if classifier_pool == 'clstok' else 0)
for classifier_pool in classifier_pool_opts:
    conv_mixer = dict(
        name="zoology.mixers.base_conv.BaseConv",
        kwargs={
            "l_max": input_seq_len + (1 if classifier_pool == 'clstok' else 0),
            "kernel_size": 3,
            "implicit_long_conv": True,
        }
    )
    attention_mixer = dict(
        name="zoology.mixers.attention.MHA",
        kwargs={
            "dropout": 0.1,
            "num_heads": 4
        },
    )
    mixer = ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, attention_mixer]}
    )
    model = ModelConfig(
        block_type = "TransformerBlock",
        d_model=d_model,
        n_layers=n_layers,
        sequence_mixer=attention_mixer,
        state_mixer=dict(name="zoology.mixers.mlp.MLP", kwargs={'hidden_mult': 1}),
        max_position_embeddings=max_position_embeddings,
        name=f"attention",
        classifier_pool=classifier_pool,
        **model_factory_kwargs
    )
    models.append(model)


for model in models:
    run_id=f'{model.name}-{embedder_type}-{model.classifier_pool}-L_{model.n_layers}'

    config = TrainConfig(
        run_id=run_id,
        # sweep_id=sweep_name,
        data=DataConfig(
            cache_dir=cache_dir,
            batch_size=batch_size,
            train_configs=train_configs,
            test_configs=test_configs
        ),
        model=model,
        logger=LoggerConfig(
            project_name=f"zoology-imgcls-train-L{n_layers}" if not is_eval_only else f"zoology-imgcls-eval-L{n_layers}",
            entity=entity_name,
            logger_type='wandb' if not is_eval_only else 'text'
        ),
        loss_fn='ce',
        eval_metrics=['accuracy'],
        slice_keys=["perturb_pixels", "perturb_mode"] if is_eval_only else [],
        max_epochs=100,
        learning_rate=0.001,
        early_stopping_metric="",
        eval_only=is_eval_only,
        eval_name=f'advatk' if is_eval_only else '',
        output_dir=f'{output_path}/{run_id}'
    )

    configs.append(config)

