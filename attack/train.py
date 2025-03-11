import os
from typing import List, Union, Tuple, Callable
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.utils import set_determinism, random_choice, merge_dict_list

from .data.utils import prepare_data
from .config import TrainConfig, DataConfig
from .model import Model
from .logger import WandbLogger, TextLogger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        eval_metrics: List[str] = [],
        train_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.eval_metrics = eval_metrics

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.slice_keys = slice_keys

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # forward
            logits = self.model(inputs)

            # collect auxiliary losses
            auxiliary_loss = []

            def get_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    auxiliary_loss.append(module.get_auxiliary_loss())

            self.model.apply(get_auxiliary_loss)
            auxiliary_loss = sum(auxiliary_loss)

            # need to flatten batch and sequence dimensions
            main_loss = self.loss_fn(logits, targets)
            loss = main_loss + auxiliary_loss
            loss.backward()
            self.optimizer.step()

            # logging and printing
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log(
                {
                    "train/loss": loss,
                    "train/main_loss": main_loss,
                    "train/auxiliary_loss": auxiliary_loss,
                    "epoch": epoch_idx,
                }
            )

    def test(self, epoch_idx: int):
        self.model.eval()

        test_loss = 0
        sample_inputs = []
        sample_preds = []
        sample_targets = []
        num_vis_output = self.logger.num_vis_output
        sample_num_per_batch = -1
        results = []

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx + 1}/{self.max_epochs}",
            postfix={key: "-" for key in (["loss"] + self.eval_metrics)},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)

                loss = self.loss_fn(logits, targets)
                test_loss += loss / len(self.test_dataloader)

                metric_result = []
                for metric in self.eval_metrics:
                    if metric == 'accuracy':
                        preds = torch.argmax(logits, dim=-1).cpu()
                        metric_result = merge_dict_list(metric_result, compute_metrics_acc(preds, targets.cpu(), slices))
                    elif metric == 'psnr':
                        metric_result = merge_dict_list(metric_result, compute_metrics_psnr(logits.cpu(), targets.cpu(), slices))
                results.extend(metric_result)

                if num_vis_output > 0:
                    if sample_num_per_batch < 0:
                        # calculate the number of samples to draw per batch
                        sample_num_per_batch = max(1, num_vis_output // len(self.test_dataloader))
                    
                    vis_sample_idx = random_choice(inputs.shape[0], sample_num_per_batch)
                    sample_inputs.append(inputs[vis_sample_idx])
                    sample_preds.append(logits[vis_sample_idx])
                    sample_targets.append(targets[vis_sample_idx])
               
                iterator.update(1)

            results = pd.DataFrame(results)

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
            }
            for metric_key in self.eval_metrics:
                avg_metric = results[metric_key].mean()
                metrics.update({f"valid/{metric_key}": avg_metric.item()})

            # compute metrics for slices
            for key in self.slice_keys:
                for metric in self.eval_metrics:
                    metric_by_slice = results.groupby(key)[metric].mean()
                    for value, metric_val in metric_by_slice.items():
                        metrics[f"valid/{key}/{metric}-{value}"] = metric_val

            iterator.set_postfix(metrics)
            self.logger.log({"epoch": epoch_idx, **metrics})
            
            if num_vis_output > 0:
                sample_inputs = torch.concat(sample_inputs, dim=0)[:num_vis_output]
                sample_preds = torch.concat(sample_preds, dim=0)[:num_vis_output]
                sample_targets = torch.concat(sample_targets, dim=0)[:num_vis_output]
                sample_log_outputs = {
                    'input': sample_inputs,
                    'pred': sample_preds,
                    'target': sample_targets
                }
                self.logger.log_output({"epoch": epoch_idx, **sample_log_outputs})

        return metrics
            
    def fit(self):
        self.model.to("cuda")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            self.logger.save_checkpoint(epoch_idx, self.model, self.optimizer)

            metrics = self.test(epoch_idx)

            # early stopping
            if self.early_stopping_metric and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break

            self.scheduler.step()
        
        self.logger.save_checkpoint(epoch_idx, self.model, self.optimizer, 'last')

    def eval(self, name=''):
        self.model.to("cuda")
        
        epoch = self.logger.restore_checkpoint(self.model, None)

        metrics = self.test(epoch)
        postfix = f'_{name}' if name else ''
        with open(os.path.join(self.logger.output_dir, f'eval_{epoch:04d}{postfix}.txt'), 'a+') as f_handle:
            for k in metrics:
                f_handle.write(f'{k}={metrics[k]} \n')            


def compute_metrics_acc(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def mse2psnr(x):
    if isinstance(x, float):
        x = torch.tensor([x])
    return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))


def compute_metrics_psnr(
    preds: torch.Tensor, 
    targets: torch.Tensor,
    slices: List[dict]
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        mse = ((pred - target) ** 2).mean()
        results.append(
            {
                "psnr": mse2psnr(mse).item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    # TODO (SE): need to actaully verify reproducibility here
    set_determinism(config.seed)
    
    if config.logger.logger_type == 'wandb':
        logger = WandbLogger(config)
    elif config.logger.logger_type == 'text':
        logger = TextLogger(config)
    logger.log_config(config)
    config.print()

    train_dataloader, test_dataloader = prepare_data(config.data)
    
    model = Model(config=config.model)
    
    if config.loss_fn == 'mse':
        loss_fn = nn.MSELoss()
    elif config.loss_fn == 'ce':
        loss_fn = lambda logits, targets: F.cross_entropy(rearrange(logits, "... c -> (...) c"), targets.flatten())
    else:
        raise NotImplementedError(f'Unsupported loss function: {config.loss_fn}')
    
    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        eval_metrics=config.eval_metrics,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger
    )
    if config.eval_only:
        task.eval(config.eval_name)
    else:
        task.fit()
        task.eval(config.eval_name)
    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train()
