import os
from pathlib import Path

import wandb
from torch.nn import Module

from .model import LanguageModel
from .config import LoggerConfig, TrainConfig

class WandbLogger:
    def __init__(self, config: TrainConfig):
        if config.logger.project_name is None or config.logger.entity is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return
        self.no_logger = False
        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name, 
        )
        # wandb.run.log_code(
        #     root=str(Path(__file__).parent.parent),
        #     include_fn=lambda path, root: path.endswith(".py")
        # )

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(
        self, 
        model: LanguageModel,
        config: TrainConfig
    ):
        if self.no_logger:
            return
        
        max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
        wandb.log(
            {
                "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "state_size": model.state_size(sequence_length=max_seq_len),
            }
        )
        wandb.watch(model)

    def log(self, metrics: dict):
        if self.no_logger:
            return
        wandb.log(metrics)
    
    def finish(self):
        if self.no_logger:
            return
        self.run.finish()



class TextLogger:
    def __init__(self, config: TrainConfig, eval_only: bool, root_path=None):
        
        if root_path is None:
            root_path = os.path.join(config.logger.root_path, config.logger.project_name)

        self.output_dir = os.path.join(root_path, config.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # create and clear files
        if not eval_only:
            for file_name in ['config.txt', 'model.txt', 'train_log.txt']:
                with open(os.path.join(self.output_dir, file_name), 'w') as f:
                    pass

    def write_dict_to_file(self, dict_to_write, file_name, splitter=''):
        with open(os.path.join(self.output_dir, file_name), 'a') as f:
            fields = []
            for key, value in dict_to_write.items():
                fields.append(f"{key}={value}")
            line = splitter.join(fields)
            print(line, file=f)

    def log_config(self, config: TrainConfig):
        self.write_dict_to_file(
            config.model_dump(),
            'config.txt',
            splitter='\n'
        )


    def log_model(
        self, 
        model: LanguageModel,
        config: TrainConfig
    ):
        max_seq_len = max([c.input_seq_len for c in config.data.test_configs])

        self.write_dict_to_file(
            {
                "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "state_size": model.state_size(sequence_length=max_seq_len),
            },
            'model.txt',
            splitter='\n'
        )
            

    def log(self, metrics: dict):

        self.write_dict_to_file(
            metrics,
            'train_log.txt',
            splitter=','
        )
    
    def finish(self):
        pass
