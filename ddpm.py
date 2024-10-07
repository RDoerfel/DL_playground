# %%
# Implementation of the denoising diffusion probabilistic model (DDPM) in PyTorch.
import os
from pathlib import Path
import torch
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from utils import setup_logging_dirs
from utils import get_mnist_data
from utils import save_images
from utils import transform_sampled_image
from ddpm_unet import LightweightUNet


# %% Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")


# %% Build diffusion related functions
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = img_size

        # set coefficients for noise schedule
        self.beta = self._prepare_noise_schedule().to(
            device
        )  # forward process variances (here we use a linear schedule)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _prepare_noise_schedule(self):
        # linear schedule for beta
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _sample_q(self, x, t):
        # Eq. 4 in the DDPM paper (Ho et al. 2020) after reparametrization
        # noisy images at time step t xt = x* sqrt(apha_bar) + sqrt(1-alpha_bar) * e
        # with e ~ N(0, I).

        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]

        # sample noise from standard normal distribution
        e = torch.randn(x.shape, device=self.device)

        # with reparametrization trick (q(xt|x0) = N(xt; x0*sqrt(alpha_bar), sqrt(1-alpha_bar)) we get the following
        # sample from q(xt|x0) = x0*sqrt(alpha_bar) + sqrt(1-alpha_bar) * e

        q_mean = sqrt_alpha_bar * x
        q_std = sqrt_one_minus_alpha_bar
        q_t = q_mean + q_std * e
        return q_t, e

    def _sample_timesteps(self, n):
        # return random integers between 1 and noise_steps of size n
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @staticmethod
    def loss_simple(epsilon, epsilon_pred) -> torch.nn.MSELoss:
        # Eq. 14 in the DDPM paper - very simple loss aka MSE between predicted and true noise
        loss = torch.nn.MSELoss()
        return loss(epsilon, epsilon_pred)

    def perform_training_step(self, model, image):
        # Step 2, Algorithm 1: sample image x0 ~q(x0)
        x_0 = image.to(self.device)
        # Step 3, Algorithm 1: generate random timesteps
        t = self._sample_timesteps(x_0.shape[0]).to(self.device)
        # Step 4, Algorithm 1: sample noise for each timestep (epsilon)
        x_t, epsilon = self._sample_q(x_0, t)

        # Step 5, Algorithm 1: Take gradient descent step on
        # predict noise
        epsilon_pred = model.forward(x_t, t)

        # calculate loss (here we use a simple MSE loss)
        loss = self.loss_simple(epsilon, epsilon_pred)

        return loss

    @torch.no_grad()
    def sample_step(self, model, x_t, t):
        # Step 3, Algorithm 2: sample noise for next time step
        # Forward Pass (predict noise for current time step)
        predicted_noise = model(x_t, t)

        # get alpha and beta for current time step
        alpha = self.alpha[t][:, None, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        # Step 3, Algorithm 2: sample noise for next time step
        if t[0] > 1:
            z_noise = torch.randn_like(x_t)
        else:
            # last step, add no noise, otherwise it would get worse
            z_noise = torch.zeros_like(x_t)

        # Step 4, Algorithm 2: update x_t-1 (remove a little bit of noise)
        x_t_minus_1 = (
            1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise)
            + torch.sqrt(beta) * z_noise
        )

        return x_t_minus_1

    def sample(self, model, t_sample_times=None):
        # Following Algorithm 2 in the DDPM paper
        model.eval()
        # Step 1, Algorithm 2:
        # start with random noise (x_T)
        x_t = torch.randn(1, 1, self.img_size, self.img_size, device=self.device)

        # Step 2, Algorithm 2: go over all time steps in reverse order (from noise to image)
        sample_images = []

        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            # timestep encoding
            t = (torch.ones(1) * i).long().to(self.device)

            # Perform a sampling step
            x_t = self.sample_step(model, x_t, t)

            # save sampled images if requested:
            if t_sample_times and i in t_sample_times:
                sample_images.append(x_t)

        model.train()

        rescaled_images = [transform_sampled_image(image) for image in sample_images]
        return rescaled_images


def train(run_name, device, epochs, lr, batch_size, image_size, dataset_path):
    ## Following Algorithm 1 in the DDPM paper
    # setup basic settings
    setup_logging_dirs(run_name)
    data_loader = get_mnist_data(image_size=image_size, batch_size=batch_size, data_path=dataset_path)
    model = LightweightUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    diffusion = Diffusion(img_size=image_size, device=device)
    logger = SummaryWriter(log_dir=os.path.join("runs", run_name))
    length = len(data_loader)

    # itereate over epochs
    for epoch in range(epochs):
        logging.info(f"Starting Epoch {epoch}")
        pbar = tqdm(data_loader, position=0)

        for i, (image, _) in enumerate(pbar):

            # perform diffusion step
            loss = diffusion.perform_training_step(model, image)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            pbar.set_postfix({"Loss": loss.item()})
            logger.add_scalar("Loss", loss.item(), epoch * length + i)

        # save model
        torch.save(model.state_dict(), os.path.join("models", run_name, f"model_{epoch}.pt"))

        # sample some images and log them
        t_sample_times = [1, 50, 100, 150, 200, 600, 800, 999]
        sampled_diffusion_steps = diffusion.sample(model, t_sample_times=t_sample_times)
        save_images(sampled_diffusion_steps, os.path.join("results", run_name, f"diffusion_steps_{epoch}.png"))
        sampled_images = []
        for i in range(16):
            sampled_images.append(diffusion.sample(model, t_sample_times=[1])[t_sample_times[-1]])
        save_images(sampled_images, os.path.join("results", run_name, f"sampled_images_{epoch}.png"))


def sample(model_path, run_name, device, image_size, t_sample_times=None):
    logging.info(f"Starting sampling at time points {t_sample_times}")
    # function to sample from a trained model
    # 1. initialize diffusion model
    diffusion = Diffusion(img_size=image_size, device=device)  # Assuming image size is 28 for MNIST

    # 2. initialize model
    model = LightweightUNet().to(device)

    # 3. load model state
    model_path = Path(model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. generate samples
    sampled_digits = diffusion.sample(model, t_sample_times=t_sample_times)

    # 5. save generated samples
    save_images(sampled_digits, os.path.join("results", run_name, f"diffusion_steps.png"))


def launch():
    import argparse

    # Example usage:
    # python ddpm.py --run_name "ddpm_run" --device "cpu" --image_size 28 train --epochs 1 --lr 0.001 --batch_size 1 --dataset_path "data"
    # python ddpm.py --run_name "ddpm_run" --device "cpu" --image_size 28 sample --model_path "./models/ddpm_run/model_0.pt" --t_sample_times 999 1

    parser = argparse.ArgumentParser(description="DDPM CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands: train or sample")

    # Common arguments
    parser.add_argument("--run_name", type=str, default="ddpm_run", help="Name of the run")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--image_size", type=int, default=28, help="Image size")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--dataset_path", type=str, default="data", help="Path to dataset")

    # Subparser for sampling
    sample_parser = subparsers.add_parser("sample", help="Sample from the trained model")
    sample_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    sample_parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to generate")
    sample_parser.add_argument(
        "--t_sample_times", type=int, nargs="+", default=None, help="List of time steps for sampling"
    )

    args = parser.parse_args()

    if args.command == "train":
        print(
            f"Running training with following args: {args.run_name} , {args.device}, {args.epochs}, {args.lr}, {args.batch_size}, {args.image_size}, {args.dataset_path}"
        )
        train(
            run_name=args.run_name,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            image_size=args.image_size,
            dataset_path=args.dataset_path,
        )
    elif args.command == "sample":
        print(
            f"Running sampling with following args: {args.run_name} , {args.device}, {args.model_path}, {args.n_samples}, {args.t_sample_times}"
        )
        sample(
            image_size=args.image_size,
            run_name=args.run_name,
            device=args.device,
            model_path=args.model_path,
            t_sample_times=args.t_sample_times,
        )


# %%
if __name__ == "__main__":
    torch.cuda.empty_cache()
    launch()

# %%
