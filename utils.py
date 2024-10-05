import torch
import torchvision
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
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_mnist_data(image_size, batch_size, data_path):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_size + int(0.25 * image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def transform_sampled_image(image):
    image = (image.clamp(-1, 1) + 1) / 2  # rescale to [0, 1]
    image = (image * 255).type(torch.uint8)  # rescale to [0, 255]
    return image
