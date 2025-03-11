import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Union

from ..config import DataSegmentConfig
from .utils import DataSegment

from zoology.utils import bchw2seq


class ImageSegmentConfig(DataSegmentConfig):
    name: str = "image_data"
    # Dataset name
    dataset: str = "mnist"
    # Path to load/download dataset
    data_path: str = "./datasets/"
    # If True, load training set
    is_trainset: bool = True
    # If True, image pixels are converted to
    # gray-scale 8-bit token indices
    is_token: bool = False

    def build(self, seed: int) -> DataSegment:
        pass


def load_pytorch_builtin_dataset(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    name: str,
    dataset: str,
    data_path: str,
    is_trainset: bool,
    seed: int,
    is_token: bool,
    **kwargs
):
    np.random.seed(seed)

    image_size = int(input_seq_len ** 0.5)
    assert image_size ** 2 == input_seq_len, "input_seq_len should be image_size x image_size"

    image_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    if is_token:
        image_transforms += [transforms.Grayscale()]
    transform = transforms.Compose(image_transforms)

    if dataset == 'cifar10':
        dataset_cls = torchvision.datasets.CIFAR10
        num_ch = 3
    elif dataset == 'mnist':
        dataset_cls = torchvision.datasets.MNIST
        num_ch = 1
    if not is_token:
        assert num_ch == vocab_size, "vocab_size should be the number of image channels"

    img_dataset = dataset_cls(root=data_path, train=is_trainset, download=True, transform=transform)
    imgs = img_dataset.data

    # Generate random image indices
    example_idx = np.arange(len(img_dataset))
    np.random.shuffle(example_idx)
    example_idx = example_idx[:num_examples]

    examples = [img_dataset[img_idx] for img_idx in example_idx]
    example_imgs, example_labels = list(zip(*examples))
    example_imgs = torch.stack(example_imgs, dim=0)
    example_labels = torch.Tensor(example_labels).long()  # (B,)

    if example_imgs.ndim == 3:
        # For single-channel images: (B, H, W) -> (B, 1, H, W)
        example_imgs = example_imgs.unsqueeze(1)
    
    if is_token:
        example_imgs = (example_imgs * 255).long()
        
    return example_imgs, example_labels

