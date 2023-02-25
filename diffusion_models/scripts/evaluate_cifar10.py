"""Script for evaluating a model trained on the CIFAR10 dataset."""

import logging
from typing import Callable, Sequence

from torch_ema import ExponentialMovingAverage

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor

from diffusion_models.metrics.inception_scores import init_inception_model, compute_inception_scores
from diffusion_models.metrics.metrics import get_bits_per_dim, get_classifier_metrics
from diffusion_models.models.ddpm import DDPM
from diffusion_models.modules.unets.ddpm_unet import DDPMUnet
from diffusion_models.modules.unets.unet import Unet
from diffusion_models.variance_schedules.variance_schedules import get_var_sched_fn
from diffusion_models.utils.config import DDPMEvalConfig, parse_config
from diffusion_models.utils.utils import load_checkpoint


logger = logging.getLogger(__name__)


def evaluate_cifar10_model(config: DDPMEvalConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])

    training_data = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=transforms
    )

    validation_data = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=True,
        transform=transforms
    )

    var_sched_fn = get_var_sched_fn(config.variance_schedule)
    noise_model = Unet(
        dim=config.image_size,
        dim_mults=(1, 2, 4, 8),
        channels=config.channels,
    )
    if config.use_ddpm_unet:
        noise_model = DDPMUnet(
            dim=config.image_size,
            dim_mults=(1, 2, 4, 8),
            image_size=config.image_size,
            channels=config.channels,
        )
    model = DDPM(
        noise_model,
        config.timesteps,
        var_sched_fn,
    )

    if config.use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay)
        ema.to(device)
    else:
        ema = None

    # Load from checkpoint.
    model, ema, ckpt_epoch = load_checkpoint(model, ema, None, config.load_from_checkpoint, train=False)

    inception_model, inception_preprocess = init_inception_model()

    metrics_list = [metric.strip().lower() for metric in config.metrics.split(',')]

    # Precompute Inception V3 logits and visual features for the Inception
    # score and FID score for the training and validation sets.
    cached_train_feats = None
    if "fid_train" in metrics_list:
        inception_model.to(device)

        train_precompute_dataloader = DataLoader(training_data, batch_size=config.precompute_batch_size)

        _, cached_train_feats = compute_inception_scores(
            train_precompute_dataloader,
            inception_model,
            inception_preprocess,
            device,
            name="train",
        )

    cached_valid_feats = None
    if "fid_valid" in metrics_list:
        inception_model.to(device)

        valid_precompute_dataloader = DataLoader(validation_data, batch_size=config.precompute_batch_size)
        
        _, cached_valid_feats = compute_inception_scores(
            valid_precompute_dataloader,
            inception_model,
            inception_preprocess,
            device,
            name="valid",
        )

    inception_model.cpu()

    # Create dataloaders for calculating the BPD, if necessary.
    train_bpd_dataloader = None
    if "bpd_train" in metrics_list:
        train_bpd_dataloader = DataLoader(training_data, batch_size=config.bpd_batch_size)
    
    test_bpd_dataloader = None
    if "bpd_test" in metrics_list:
        test_bpd_dataloader = DataLoader(validation_data, batch_size=config.bpd_batch_size)

    metrics_dict = {metric.lower(): None for metric in metrics_list}

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
        logger.info(f"Epoch {ckpt_epoch} {metric}: {metric_val}")


def main():
    # Define the logger...?
    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
        level=logging.INFO,
    )

    eval_config = parse_config(DDPMEvalConfig)

    evaluate_cifar10_model(eval_config)


if __name__ == '__main__':
    main()