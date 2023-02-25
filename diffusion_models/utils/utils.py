"""Utilities for our VDM implementation."""

import logging
import os
from inspect import isfunction

import torch


logger = logging.getLogger(__name__)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def save_checkpoint(epoch, model, ema, optim, ckpt_path):
    """Saves a checkpoint object, which includes the current epoch, model
    state dict, EMA state dict (if available), and optimizer state dict."""
    logger.info(f"Saving checkpoint for epoch {epoch} to {ckpt_path}...")
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
    }
    if exists(ema):
        state['ema'] = ema.state_dict()
    # May not always need to save the optimizer (e.g. if it has no internal state).
    if exists(optim):
        state['optimizer'] = optim.state_dict()
    torch.save(state, ckpt_path)
    logger.info(f"Saved checkpoint for epoch {epoch}.")


def load_checkpoint(model, ema, optim, ckpt_path, train=True):
    """Loads all of the saved model components from a checkpoint file."""
    start_epoch = 0
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state'])
        if exists(ema):
            ema.load_state_dict(ckpt['ema'])
        if train and exists(optim):
            optim.load_state_dict(ckpt['optimizer'])
        logger.info(f"Loaded checkpoint at epoch {start_epoch}.")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}.")
    
    if train:
        return model, ema, optim, start_epoch
    else:
        return model, ema, start_epoch


