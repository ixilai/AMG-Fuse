"""
Helper script to load Restormer checkpoint for training.

During training, the Restormer restoration network is used for the
Task-Coupled Degradation-Aware Strategy (TDAS) to guide the fusion network.

Usage in training:
    model = MSAKNet(use_restormer=True)

    # Load weather-specific Restormer checkpoint
    restormer_ckpt = torch.load('path/to/restormer_derain.pth')  # or desnow, dehaze
    model.res.load_state_dict(restormer_ckpt["AWF"], strict=False)

    # Continue with training
"""

import torch

def load_restormer_checkpoint(model, ckpt_path):
    """
    Load pre-trained Restormer checkpoint into the model.

    Args:
        model: MSAKNet model with use_restormer=True
        ckpt_path: Path to Restormer checkpoint (derain/desnow/dehaze)
    """
    if not model.use_restormer:
        raise ValueError("Model was initialized with use_restormer=False")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.res.load_state_dict(checkpoint["AWF"], strict=False)
    print(f"Loaded Restormer checkpoint from {ckpt_path}")

    return model


# Example usage in your training script:
#
# from model.amgfuse import MSAKNet
# from model.load_restormer import load_restormer_checkpoint
#
# # Create model with Restormer
# model = MSAKNet(use_restormer=True).to(device)
#
# # Load appropriate Restormer checkpoint for your weather condition
# restormer_ckpt = 'checkpoints/restormer_derain.pth'  # or desnow, dehaze
# model = load_restormer_checkpoint(model, restormer_ckpt)
#
# # Continue with training...
