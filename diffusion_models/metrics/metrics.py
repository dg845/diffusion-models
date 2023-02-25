"""Calculate metrics for evaluating the diffusion models."""

import logging
from collections.abc import Sequence
from typing import Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from diffusion_models.metrics.classifier_metrics_numpy import classifier_score_from_logits, frechet_classifier_distance_from_activations
from diffusion_models.metrics.datasets import SampleDataset
from diffusion_models.metrics.inception_scores import compute_inception_scores
from diffusion_models.utils.utils import exists


logger = logging.getLogger(__name__)


def get_classifier_metrics(
    metrics_dict,
    config,
    gen_model,
    ema,
    classifier_model,
    classifier_preprocess,
    cached_train_feats,
    cached_valid_feats,
    device="cpu",
):
    metrics_calculated = ["inception_score", "fid_train", "fid_valid"]
    if all(metric not in metrics_dict for metric in metrics_calculated):
        # No-op
        return metrics_dict

    gen_model.eval()
    classifier_model.eval()

    using_ema = exists(ema)

    # Move generative model to device if not already there.
    gen_model.to(device)

    # If using EMA, store the original model weights and load the EMA parameters to the model.
    if using_ema:
        ema.to(device)
        ema.store()
        ema.copy_to()

    # Sample eval_batch_size generated images from the model for the Inception
    # score.
    if "inception_score" in metrics_dict:
        logger.info("Getting samples for calculating the inception score...")
        inception_dataloader = get_sample_dataloader(
            gen_model,
            config.eval_batch_size,
            min(config.sample_batch_size, config.inception_samples),
            config.inception_samples,
            config.channels,
            config.image_size,
        )
        logger.info("Finished getting inception samples.")

    if "fid_train" in metrics_dict:
        train_size = cached_train_feats.shape[0]
        logger.info("Getting samples for calculating FID training set score...")
        fid_train_gen_samples_dataloader = get_sample_dataloader(
            gen_model,
            config.eval_batch_size,
            config.sample_batch_size,
            train_size,
            config.channels,
            config.image_size,
        )
        logger.info("Finished getting FID train samples.")

    if "fid_valid" in metrics_dict:
        valid_size = cached_valid_feats.shape[0]
        logger.info("Getting samples for calculating the FID validation set scores...")
        fid_valid_gen_samples_dataloader = get_sample_dataloader(
            gen_model,
            config.eval_batch_size,
            config.sample_batch_size,
            valid_size,
            config.channels,
            config.image_size,
        )
        logger.info("Finished getting FID valid samples.")
    
    # If using ema, restore the original model weights.
    if using_ema:
        ema.restore()
        ema.to("cpu")

    # Move generative model off device to save space.
    gen_model.cpu()
    # Move classifier model to device
    classifier_model.to(device)

    if "inception_score" in metrics_dict:
        logger.info("Calculating inception scores...")
        metrics_dict["inception_score"] = calculate_inception_v3_score(
            inception_dataloader,
            classifier_model,
            classifier_preprocess,
            device,
            name="Inception",
        )
        logger.info("Finished calculating inception scores.")

    if "fid_train" in metrics_dict:
        logger.info("Calculating FID train scores...")
        metrics_dict["fid_train"] = calculate_fid_score(
            fid_train_gen_samples_dataloader,
            classifier_model,
            classifier_preprocess,
            cached_train_feats,
            device,
            name="FID train",
        )
        logger.info("Finished calculating FID train scores.")

    if "fid_valid" in metrics_dict:
        logger.info("Calculating FID valid scores...")
        metrics_dict["fid_valid"] = calculate_fid_score(
            fid_valid_gen_samples_dataloader,
            classifier_model,
            classifier_preprocess,
            cached_valid_feats,
            device,
            name="FID valid",
        )
        logger.info("Finished calculating FID valid scores.")

    # Metrics arrays are already moved to CPU
    # Move classifier model to CPU to save space.
    classifier_model.cpu()
    gen_model.to(device)
    if using_ema:
        ema.to(device)

    return metrics_dict


def calculate_inception_v3_score(
    dataloader,
    cls_model,
    cls_preprocess,
    device,
    name="Inception",
):
    logits, _ = compute_inception_scores(
        dataloader,
        cls_model,
        cls_preprocess,
        device,
        name=name,
    )

    inception_score = classifier_score_from_logits(logits)

    return inception_score


def calculate_fid_score(
    dataloader,
    cls_model,
    cls_preprocess,
    cached_features,
    device,
    name="FID",
):
    _, visual_features = compute_inception_scores(
        dataloader,
        cls_model,
        cls_preprocess,
        device,
        name=name,
    )

    fid_score = frechet_classifier_distance_from_activations(
        np.squeeze(cached_features), np.squeeze(visual_features)
    )

    return fid_score


def get_batch_list(num_samples: int, batch_size: int):
    num_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    batch_list = [batch_size] * num_batches

    if remainder:
        batch_list.append(remainder)
    
    return batch_list


def get_model_samples(
    model,
    num_samples: int,
    batch_size: int,
    channels: int,
    img_size: int,
):
    samples = []
    batch_list = get_batch_list(num_samples, batch_size)
    for batch in tqdm(
        batch_list,
        desc="Sample batches",
        total=len(batch_list)
    ):
        sample = model.sample(batch_size=batch, channels=channels, image_size=img_size)
        sample = sample[-1]
        samples.append(sample)
    
    samples = torch.cat(samples, axis=0)

    return samples


def get_sample_dataloader(
    model,
    eval_batch_size: int,
    sample_batch_size: int,
    num_samples: int,
    channels: int,
    img_size: int
):
    samples = get_model_samples(model, num_samples, sample_batch_size, channels, img_size)
    sample_dataset = SampleDataset(samples)
    sample_dataloader = DataLoader(sample_dataset, batch_size=eval_batch_size)

    return sample_dataloader

def get_bits_per_dim(
    metrics_dict,
    gen_model,
    ema,
    train_dataloader,
    test_dataloader,
    device,
):
    """Calculates (an estimate of) the negative log likelihood, expressed in
    bits per dimesnion.
    """
    metrics_calculated = ["bpd_train", "bpd_test"]
    if all(metric not in metrics_dict for metric in metrics_calculated):
        # No-op
        return metrics_dict
    
    using_ema = exists(ema)

    gen_model.to(device)
    if using_ema:
        ema.to(device)
        ema.store()
        ema.copy_to()

    if "bpd_train" in metrics_dict:
        train_bpd = calculate_bits_per_dim(train_dataloader, gen_model, device, name="train")
        metrics_dict["bpd_train"] = train_bpd
    
    if "bpd_test" in metrics_dict:
        test_bpd = calculate_bits_per_dim(test_dataloader, gen_model, device, name="test")
        metrics_dict["bpd_test"] = test_bpd
    
    if using_ema:
        ema.restore()

    return metrics_dict

def calculate_bits_per_dim(dataloader, gen_model, device, name="train"):
    batch_bpds = []

    for batch in tqdm(
        dataloader,
        desc=f"Calculating {name} bits / dim",
        total=len(dataloader)
    ):
        if isinstance(batch, Sequence):
            batch = batch[0]
        
        batch = batch.to(device)

        batch_bpd_b = gen_model.calc_bpd_loop(batch, device=device)

        batch_bpds.append(batch_bpd_b)
    
    bpd = torch.mean(torch.cat(batch_bpds)).cpu().item()

    return bpd
