"""Script for training a diffusion model on the CIFAR10 dataset."""

import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor


from diffusion_models.metrics.inception_scores import init_inception_model, compute_inception_scores
from diffusion_models.metrics.metrics import get_bits_per_dim, get_classifier_metrics
from diffusion_models.models.ddpm import DDPM
from diffusion_models.losses.vdm_loss import p_losses
from diffusion_models.modules.unets.ddpm_unet import DDPMUnet
from diffusion_models.modules.unets.unet import Unet
from diffusion_models.variance_schedules.variance_schedules import get_var_sched_fn
from diffusion_models.utils.config import DDPMConfig, parse_config
from diffusion_models.utils.sample import make_grid
from diffusion_models.utils.utils import exists, load_checkpoint, save_checkpoint


logger = logging.getLogger(__name__)


def plot_samples(
    epoch,
    model,
    ema,
    reverse_transform,
    results_folder: Path,
    sqrt_num_samples: int=10,
    channels: int=1,
    img_size: int=28,
    mode: str="RGB",
):
    model.eval()
    if exists(ema):
        ema.store()
        ema.copy_to()
    
    logger.info("Getting samples for visualization...")
    samples = model.sample(batch_size=sqrt_num_samples ** 2, channels=channels, image_size=img_size)
    denoised_samples = samples[-1]
    denoised_samples = reverse_transform(denoised_samples)
    # Convert to numpy
    # numpy_samples = denoised_samples.numpy().round().astype(np.uint8)
    logger.info("Finished getting samples.")

    if exists(ema):
        ema.restore()
    
    # Convert to PIL images; assume images is in shape BHWC
    if denoised_samples.shape[-1] == 1:
        pil_samples = [Image.fromarray(image.squeeze(), mode="L") for image in denoised_samples]
        image_grid = make_grid(pil_samples, sqrt_num_samples, sqrt_num_samples, mode="L")
    else:
        pil_samples = [Image.fromarray(image) for image in denoised_samples]
        image_grid = make_grid(pil_samples, sqrt_num_samples, sqrt_num_samples, mode=mode)
    
    sample_path = results_folder / f"sample-{epoch}.png"
    logger.info(f"Saving samples for epoch {epoch} to {sample_path}...")
    image_grid.save(sample_path)

    # return image_grid, denoised_samples


def train_one_epoch(
    epoch: int,
    dataloader: DataLoader,
    model,
    optimizer,
    ema,
    device,
    writer,
    timesteps: int=200,
    loss_type: str="huber",
):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"Epoch {epoch} Steps")
    for step, (batch, _) in enumerate(dataloader):
        batch_size = batch.shape[0]
        batch = batch.to(device)

        # Sample timesteps uniformly at random from [0,..., T - 1] each sample
        # in the batch.
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # The model forward pass occurs in the loss function.
        loss = model.p_losses(batch, t, loss_type=loss_type, device=device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update the parameters via EMA, if applicable.
        if exists(ema):
            ema.update()
        
        step_loss = loss.detach().item()
        epoch_loss += step_loss

        # Print out the training loss, hardcode the interval (100) for now.
        # if step % 100 == 0:
        #     logger.info(f"\nLoss at epoch {epoch}, step {step}: {step_loss}")
        
        logs = {"step_loss": step_loss}
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)
    
    epoch_loss /= step
    writer.add_scalar('metrics/training_loss', epoch_loss, epoch)
    # logger.info(f"\nLoss at epoch {epoch}: {epoch_loss}")

    return epoch_loss


def train_diffusion_model(config: DDPMConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a SummaryWriter to write out data to tensorboard.
    writer = SummaryWriter(logdir=config.logdir) 

    # ----Get dataset and dataloader for built-in CIFAR10 dataset----
    
    # Original DDPM paper uses random horizontal flips.
    # Go from PIL NumPy array in {0,...,255} to PyTorch tensor in [-1, 1].
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])
    # Reverse transforms to go back to a PIL image from a PyTorch tensor image.
    reverse_transforms = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(0, 2, 3, 1)), # Convert from BCHW to BHWC
        Lambda(lambda t: t * 255),
        Lambda(lambda t: t.numpy().round().astype(np.uint8)),
    ])

    training_data = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=transforms,
    )
    train_dataloader = DataLoader(training_data, batch_size=config.train_batch_size, shuffle=True)

    if config.perform_eval:
        validation_data = datasets.CIFAR10(
            root=config.data_root,
            train=False,
            download=True,
            transform=transforms,
        )
        valid_dataloader = DataLoader(validation_data, batch_size=config.eval_batch_size)

    channels = config.channels
    img_size = config.image_size

    # ----Define model (hardcoded to Unet for now)----

    # Match the DDPM model defined in
    # https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py
    # as closely as possible.
    # noise_model = Unet(
    #     dim=128,
    #     dim_mults=(1, 2, 2, 2),
    #     channels=channels,
    # )
    noise_model = Unet(
        dim=img_size,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
    )
    if config.use_ddpm_unet:
        noise_model = DDPMUnet(
            dim=img_size,
            dim_mults=(1, 2, 4, 8),
            image_size=img_size,
            channels=channels,
        )
    var_sched_fn = get_var_sched_fn(config.variance_schedule)
    model = DDPM(
        noise_model,
        config.timesteps,
        var_sched_fn,
    )
    model.to(device)

    # ----Define optimizer (hardcoded to Adam for now)----
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # ----Initialize the EMA wrapper, if using EMA----
    if config.use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay)
        ema.to(device)
    else:
        ema = None

    # ----Define loss function----
    # loss_fn = p_losses

    # ----Saving and Loading from Checkpoints----
    checkpoints_folder = Path(config.checkpoints_folder)
    checkpoints_folder.mkdir(exist_ok=True)

    # Load from checkpoint.
    start_epoch = 0
    if exists(config.load_from_checkpoint):
        model, ema, optimizer, start_epoch = load_checkpoint(
            model, ema, optimizer, config.load_from_checkpoint, train=True
        )
        logger.info(f"Restarting training from epoch {start_epoch}.")

    # ----Set up evaluation----

    # Saving samples
    # Hardcode path for now
    samples_folder = Path(config.samples_folder)
    samples_folder.mkdir(exist_ok=True)

    if config.perform_eval:
        metrics_list = [metric.strip().lower() for metric in config.metrics.split(',')]
        inception_model, inception_preprocess = init_inception_model()

        cached_train_feats = None
        cached_valid_feats = None

        # Precompute Inception V3 logits and visual features for the Inception
        # score and FID score for the training and validation sets.
        if "fid_train" in metrics_list:
            inception_model.to(device)

            # Create a new dataloader on the training set using eval_batch_size.
            train_eval_dataloader = DataLoader(training_data, batch_size=config.eval_batch_size)

            _, cached_train_feats = compute_inception_scores(
                train_eval_dataloader,
                inception_model,
                inception_preprocess,
                device,
            )
        
        if "fid_valid" in metrics_list:
            inception_model.to(device)

            _, cached_valid_feats = compute_inception_scores(
                valid_dataloader,
                inception_model,
                inception_preprocess,
                device,
            )

        inception_model.to("cpu")

        # Create dataloaders for calculating the BPD, if necessary.
        train_bpd_dataloader = None
        if "bpd_train" in metrics_list:
            train_bpd_dataloader = DataLoader(training_data, batch_size=config.bpd_batch_size)
        
        test_bpd_dataloader = None
        if "bpd_test" in metrics_list:
            test_bpd_dataloader = DataLoader(validation_data, batch_size=config.bpd_batch_size)

        metrics_dict = {metric.lower(): None for metric in metrics_list}

    # ----Training loop----
    progress_bar = tqdm(range(start_epoch, start_epoch + config.epochs))
    progress_bar.set_description("Epochs")
    for epoch in range(start_epoch, start_epoch + config.epochs):
        # Put model on device, if not already there.
        model.to(device)
        if config.use_ema:
            ema.to(device)
        
        epoch_loss = train_one_epoch(
            epoch + 1,
            train_dataloader,
            model,
            optimizer,
            ema,
            device,
            writer,
            timesteps=config.timesteps,
            loss_type=config.loss_type,
        )

        # Save checkpoints every save_checkpoint_epochs.
        if (epoch + 1) % config.save_checkpoint_epochs == 0:
            ckpt_path = checkpoints_folder / f"cifar10-ckpt-{epoch + 1}.pth"
            save_checkpoint(epoch + 1, model, ema, optimizer, ckpt_path)

        # Plot samples every save_samples_epochs.
        if (epoch + 1) % config.save_samples_epochs == 0:
            plot_samples(
                epoch + 1,
                model,
                ema,
                reverse_transforms,
                samples_folder,
                sqrt_num_samples=8,
                channels=channels,
                img_size=img_size,
            )

            # Write out images to tensorboard.
            # sample_image = torchvision.utils.make_grid(
            #     torch.tensor(sample_images_numpy, dtype=torch.uint8, device="cpu"),
            #     nrow=8,
            #     # value_range=(0, 255),
            # )
            # writer.add_image('Samples', sample_image, epoch + 1, dataformats="HWC")

        # Evaluate the model every eval_epochs.
        if config.perform_eval and (epoch + 1) % config.eval_epochs == 0:
            metrics_dict = get_classifier_metrics(
                metrics_dict,
                config,
                model,
                ema,
                inception_model,
                inception_preprocess,
                cached_train_feats,
                cached_valid_feats,
                device=device,
            )

            metrics_dict = get_bits_per_dim(
                metrics_dict,
                model,
                ema,
                train_bpd_dataloader,
                test_bpd_dataloader,
                device,
            )

            for metric, metric_val in metrics_dict.items():
                writer.add_scalar(f"metrics/{metric}", metric_val, epoch + 1)
                logger.info(f"Epoch {epoch + 1} {metric}: {metric_val}")
        
        logs = {"epoch_loss": epoch_loss}
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)

    writer.close()


def main():
    # Define the logger
    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
        level=logging.INFO,
    )

    config = parse_config(DDPMConfig)

    train_diffusion_model(config)


if __name__ == '__main__':
    main()