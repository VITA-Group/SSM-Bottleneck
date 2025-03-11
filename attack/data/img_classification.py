import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Union

from ..config import DataSegmentConfig
from .utils import DataSegment
from .image_data import ImageSegmentConfig, load_pytorch_builtin_dataset
from zoology.utils import bchw2seq


uniform_sample = lambda a, n: a[torch.multinomial(torch.ones_like(a).float(), n, replacement=True)]

class ImageClassificationConfig(ImageSegmentConfig):
    name: str = "image_classification"
    
    # Indices of pixels that will be perturbated
    perturb_pixels: List[int] = [0, 0]
    # Perturbation pattern: 'zeors' / 'rand' / 'addrand' / 'target' / 'none'
    perturb_mode: str = 'none'

    # Perturbation label for target attack
    perturb_target_label: int = 0

    def build(self, seed: int) -> DataSegment:
        print('Building dataset:', self.dataset)

        if self.dataset in ['mnist', 'cifar10']:
            input_imgs, target_labels = load_pytorch_builtin_dataset(**self.model_dump(), seed=seed)
        else:
            raise NotImplementedError()

        input_imgs = bchw2seq(input_imgs)  # (B, H * W, C)
        if self.is_token:
            input_imgs = input_imgs[:, :, 0]   # (B, H * W)

        if self.perturb_pixels:
            p0, p1 = self.perturb_pixels
            if self.perturb_mode == 'zeros':
                input_imgs[:, p0:p1] = torch.zeros_like(input_imgs[:, p0:p1])
            elif self.perturb_mode == 'randn' and not self.is_token:
                input_imgs[:, p0:p1] = torch.randn_like(input_imgs[:, p0:p1])
            elif self.perturb_mode == 'addrandn' and not self.is_token:
                input_imgs[:, p0:p1] += torch.randn_like(input_imgs[:, p0:p1])
            elif self.perturb_mode == 'target':
                target_mask = target_labels == self.perturb_target_label
                target_idx = torch.nonzero(target_mask).squeeze()
                other_mask = ~target_mask
                other_idx = torch.nonzero(other_mask).squeeze()

                resample_idx = uniform_sample(target_idx, other_mask.sum())

                target_labels[other_idx] = target_labels[resample_idx]
                input_imgs[other_idx, p0:p1] = input_imgs[resample_idx, p0:p1]

                target_labels = target_labels[other_idx]
                input_imgs = input_imgs[other_idx]


        return DataSegment(
            input_imgs, 
            target_labels, 
            slices={"input_seq_len": self.input_seq_len,
                    "perturb_pixels": ':'.join([str(i) for i in self.perturb_pixels]),
                    'perturb_mode': self.perturb_mode}
        )

