import os
import glob
import torch
from pathlib import Path
import numpy as np
import wandb
import torchvision
from torch.nn import Module

from .model import Model
from .config import LoggerConfig, TrainConfig
from zoology.utils import seq2bchw


class BaseLogger:
    def __init__(self, config:TrainConfig):
        self.num_vis_output = config.logger.num_vis_output
        self.vis_meta = config.logger.vis_meta

        self.max_num_ckpts = config.logger.max_num_ckpts

        if config.output_dir:
            self.output_dir = config.output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
    
    def save_checkpoint(self, epoch_idx, model, optimizer, identifier=''):
        if self.output_dir:
            ckpt_dict = {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_idx
            }
            ckpt_id = identifier if identifier else f'{epoch_idx:04d}'
            ckpt_path = os.path.join(self.output_dir, f'checkpoint_{ckpt_id}.pth')
            torch.save(ckpt_dict, ckpt_path)

            ckpt_ls = sorted(glob.glob(os.path.join(self.output_dir, 'checkpoint_*.pth')))
            if len(ckpt_ls) > self.max_num_ckpts:
                ckpts_to_rm = ckpt_ls[:len(ckpt_ls) - self.max_num_ckpts]
                for ckpt_rm in ckpts_to_rm:
                    os.remove(ckpt_rm)

    def restore_checkpoint(self, model, optimizer=None):
        if self.output_dir:
            ckpt_ls = sorted(glob.glob(os.path.join(self.output_dir, 'checkpoint_*.pth')))
            ckpt_to_load = ckpt_ls[-1]
            ckpt_dict = torch.load(ckpt_to_load)

            epoch = ckpt_dict['epoch']
            model.load_state_dict(ckpt_dict['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
            return epoch
        return 0


class WandbLogger(BaseLogger):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

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
        model: Model,
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
    
    def log_output(self, outputs: dict):
        if self.no_logger:
            return
        
        log_dict = {}

        vis_type = self.vis_meta.get('type', '')
        if vis_type == 'image':
            '''
            an example of vis_meta for image output:
            {
                'type': 'image',
                'chw': (3, 28, 28),
                'range': {
                    'input': [0., 1.],
                    'pred': [-1., 1.],
                    'target': [-1., 1.]
                }
            }
            '''
            chw = self.vis_meta.get('chw', (3, 1, 1))
            # range to normalize each visualization
            ranges = self.vis_meta.get('range', {
                'input': [0., 1.],
                'pred': [0., 1.],
                'target': [0., 1.]
            })
            imgs_vis_ls = []
            for key in outputs:
                if key == 'epoch':
                    continue

                imgs = outputs[key]
                imgs = seq2bchw(imgs, *chw)
                # normalize
                imgs = (imgs - ranges[key][0]) / (ranges[key][1] - ranges[key][0])
                imgs_vis = torchvision.utils.make_grid(imgs)
                imgs_vis = (imgs_vis.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                imgs_vis_ls.append(wandb.Image(imgs_vis, caption=f"{key}"))
            log_dict['output_vis_image'] = imgs_vis_ls
        else:
            return
        
        wandb.log({'epoch': outputs['epoch'], **log_dict})
    
    def finish(self):
        if self.no_logger:
            return
        self.run.finish()


class TextLogger(BaseLogger):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

    def log_config(self, config: TrainConfig):
        print(config.model_dump())

    def log_model(
        self, 
        model: Model,
        config: TrainConfig
    ):
        max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
        
        print("num_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("state_size", model.state_size(sequence_length=max_seq_len))
        print(model)

    def log(self, metrics: dict):
        # for k in metrics:
        #     print(f'{k}={metrics[k]}', end=' ')
        # print('\n')
        pass
    
    def log_output(self, outputs: dict):
        pass
    
    def finish(self):
        pass