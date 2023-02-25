"""Parsing and training configuration logic."""

import argparse
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class DDPMConfig:
    # ----Diffusion Model----
    variance_schedule: str = "linear"
    timesteps: int = 1000
    loss_type: str = "l2"

    image_size: int = 32
    channels: int = 3

    use_ddpm_unet: bool = False

    # ----Training----
    epochs: int = 50
    train_batch_size: int = 128
    learning_rate: float = 2e-4
    use_ema: bool = True
    ema_decay: float = 0.9999

    data_root: str = "./data"

    # ----Evaluation----
    perform_eval: bool = True
    eval_epochs: int = 50
    metrics: str = "inception_score"
    inception_samples: int = 256
    eval_batch_size: int = 8
    sample_batch_size: int = 256
    bpd_batch_size: int = 256
    
    save_samples_epochs: int = 50

    # ----Saving and Loading----
    load_from_checkpoint: Optional[str] = None
    save_checkpoint_epochs: int = 50
    checkpoints_folder: str = "checkpoints"
    samples_folder: str = "samples"
    logdir: str = "logs"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Parse the configuration from a JSON file. Will override any command line args.",
    )

    # ----Diffusion Model----

    parser.add_argument(
        "--variance_schedule",
        type=str,
        default="linear",
        help="The type of variance schedule used (cosine, linear, quadratic, sigmoid).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of discrete time steps for the diffusion process.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        help="Reconstruction loss type for training the denoising model (l1, l2, huber)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Resolution of (square) input images to the diffusion model.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of channels of input images.",
    )
    parser.add_argument(
        "--use_ddpm_unet",
        default=False,
        action="store_true",
        help="Whether to use the original DDPM paper U-Net as the denoising model.",
    )
    
    # ----Training----

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for training the denoising model.",
    )
    # Exponential Moving Average (EMA)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA during training.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="Decay parameter for EMA during training.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Directory which contains the training data.",
    )

    # ----Evaluation----

    parser.add_argument(
        "--perform_eval",
        action="store_true",
        help="Whether to perform evaluation during training.",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=50,
        help="Perform evaluation every X epochs.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="inception_score",
        help="Metrics to calculate (bpd_train, bpd_test, inception_score, fid_train, fid_test).",
    )
    parser.add_argument(
        "--inception_samples",
        type=int,
        default=256,
        help="Number of generated samples to use for calculating the inception score.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for calculating classifier metrics like IS and FID.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=256,
        help="Size of each batch we supply to the DDPM sample(...) method during evaluation.",
    )
    parser.add_argument(
        "--bpd_batch_size",
        type=int,
        default=256,
        help="Batch size for calculating the BPD on the train and/or test set.",
    )
    parser.add_argument(
        "--save_samples_epochs",
        type=int,
        default=50,
        help="Save samples every X epochs.",
    )
    parser.add_argument(
        "--precompute_batch_size",
        type=int,
        default=8,
        help="Batch size for Inception V3 inference when precomputing classifier scores for FID.",
    )

    # ----Saving and Loading----

    # Save checkpoints
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from, if desired.",
    )
    parser.add_argument(
        "--save_checkpoint_epochs",
        type=int,
        default=50,
        help="Save checkpoints every X epochs.",
    )
    # Output directories
    parser.add_argument(
        "--checkpoints_folder",
        type=str,
        default="checkpoints",
        help="Directory to which to save checkpoints.",
    )
    parser.add_argument(
        "--samples_folder",
        type=str,
        default="samples",
        help="Directory to which to save samples.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Directory to which to save log outputs (e.g. tensorboard files).",
    )

    args = parser.parse_args()

    return args


def parse_config(config_cls):
    args = parse_args()
    args_dict = vars(args)
    if "config" in args_dict and args_dict["config"]is not None:
        config = parse_json_to_config(args_dict["config"], config_cls)
    else:
        config = parse_arg_dict_to_config(args_dict, config_cls)
    return config


def parse_arg_dict_to_config(args_dict, config_cls):
    config_dict = dict()
    for arg, arg_val in args_dict.items():
        if hasattr(config_cls, arg):
            config_dict[arg] = arg_val
    return config_cls(**config_dict)


def parse_json_to_config(json_file, config_cls):
    with open(json_file, 'r') as json_config:
        json_dict = json.load(json_config)
    config = parse_arg_dict_to_config(json_dict, config_cls)
    return config


@dataclass
class DDPMEvalConfig:
    # ----Diffusion Model----
    variance_schedule: str = "linear"
    timesteps: int = 1000

    image_size: int = 32
    channels: int = 3

    use_ddpm_unet: bool = False

    # ----Training----
    use_ema: bool = True
    ema_decay: float = 0.9999

    data_root: str = "./data"

    # ----Evaluation----
    metrics: str = "inception_score"
    inception_samples: int = 256
    eval_batch_size: int = 8
    sample_batch_size: int = 256
    bpd_batch_size: int = 256
    precompute_batch_size: int = 8

    # ----Saving and Loading----
    load_from_checkpoint: Optional[str] = None