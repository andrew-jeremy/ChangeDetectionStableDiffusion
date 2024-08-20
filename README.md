# ChangeDetectionStableDiffusion
Diffusion Model Based Change Detection in Remote Sensing Images
# DiffusionBasedChangeDetector

## Overview

`DiffusionBasedChangeDetector` is a flexible and powerful change detection framework built using PyTorch. The model combines different change detection techniques, such as simple difference maps, VAE latent space change maps, and SSIM-based change maps. Additionally, it integrates object detection to refine the change maps by focusing on areas with identifiable objects.

## Features

- **Simple Difference Change Map**: Computes pixel-wise absolute differences between two input images.
- **VAE Latent Change Map**: Uses a Variational Autoencoder to encode images into a latent space, computing changes in that space.
- **SSIM-Based Change Map**: Leverages the Structural Similarity Index (SSIM) to focus on perceptual differences.
- **Object Detection Filtering**: Optionally refines change maps by retaining only regions where objects are detected.

## Directory Structure

- **`diffusion_based_change_detector/`**: Contains the main model code, utility functions, and training script.
  - `__init__.py`: Initializes the package.
  - `model.py`: Contains the `DiffusionBasedChangeDetector` class and related models (e.g., VAE, LatentDifferenceNetwork).
  - `utils.py`: Contains utility functions such as SSIM computation and object detection filtering.
  - `train.py`: Script for training the model with different configurations.
- **`examples/`**: Contains example usage scripts.
  - `example_usage.py`: Example of how to use the `DiffusionBasedChangeDetector` with different configurations.
- **`requirements.txt`**: Lists all dependencies required to run the model.
- **`README.md`**: Provides an overview of the project, installation instructions, and usage examples.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
