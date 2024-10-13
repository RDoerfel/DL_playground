import os
import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    images_np_array = np.concatenate([image.cpu().numpy().squeeze() for image in images], axis=1)
    img = Image.fromarray(images_np_array)
    img.save(path)


def get_mnist_data(image_size, batch_size, data_path):
    thresshold = 0.5
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), 
        torchvision.transforms.Lambda(
            lambda x: (thresshold < x).float().squeeze())
        ]
    )
    dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def transform_sampled_image(image):
    # image = (image.clamp(-1, 1) + 1) / 2  # rescale to [0, 1]
    # image = (image * 255).type(torch.uint8)  # rescale to [0, 255]
    return image


def setup_logging_dirs(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
