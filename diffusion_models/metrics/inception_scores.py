"""Gets Inception V3 scores for use in calculating the Inception score and 
FID score metrics."""

from collections.abc import Sequence

import numpy as np
from tqdm.auto import tqdm

import torch
from torchvision.models import Inception_V3_Weights, inception_v3


def init_inception_model():
    weights = Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()
    model = inception_v3(weights=weights)
    model.eval()
    return model, preprocess


def compute_inception_scores(dataloader, model, preprocess, device, name="classifier"):
    """Computes the Inception V3 avgpool intermediate output and logits for a
    batch of data (e.g. generated images)."""
    features = {}
    def get_features(name):
        """Creates a forward hook to get model intermediate layer outputs."""
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # Add forward hook to get the 'avgpool' layer outputs.
    model.avgpool.register_forward_hook(get_features('avgpool'))

    # Logits for Inception score
    logits = []
    # Visual features for FID score
    visual_features = []

    # Don't use label for now
    for batch in tqdm(
        dataloader,
        desc=f"Computing {name} scores",
        total=len(dataloader),
    ):
        if isinstance(batch, Sequence):
            batch = batch[0]
        
        # Run preprocessing transforms on CPU for now
        batch = preprocess(batch)
        batch = batch.to(device)

        # Forward pass, also extracts features due to our forward hook
        preds = model(batch)

        # Store as NumPy arrays
        logits.append(preds.detach().cpu().numpy())
        visual_features.append(features['avgpool'].cpu().numpy())
    
    # Concatenate into one big NumPy array at max precision (float64).
    logits = np.concatenate(logits, axis=0).astype(np.float64)
    visual_features = np.concatenate(visual_features, axis=0).astype(np.float64)
    
    return logits, visual_features


def precompute_inception_scores(dataloader, model, preprocess, device, name="classifier"):
    """(Pre)computes the Inception V3 avgpool intermediate output and logits
    for use in the Inception and FID score metrics."""
    features = {}
    def get_features(name):
        """Creates a forward hook to get model intermediate layer outputs."""
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # Add forward hook to get the 'avgpool' layer outputs.
    model.avgpool.register_forward_hook(get_features('avgpool'))

    # Logits for Inception score
    logits = []
    # Visual features for FID score
    visual_features = []

    # Don't use label for now
    for i, (batch, _) in tqdm(
        enumerate(dataloader),
        desc=f"Precomputing {name} scores",
        total=len(dataloader),
    ):
        # Run preprocessing transforms on CPU for now
        batch = preprocess(batch)
        batch = batch.to(device)

        # Forward pass, also extracts features due to our forward hook
        preds = model(batch)

        # Store as NumPy arrays
        logits.append(preds.detach().cpu().numpy())
        visual_features.append(features['avgpool'].cpu().numpy())
    
    # Concatenate into one big NumPy array at max precision (float64).
    logits = np.concatenate(logits, axis=0).astype(np.float64)
    visual_features = np.concatenate(visual_features, axis=0).astype(np.float64)
    
    return logits, visual_features


