'''
Description:
This script contains the main model architecture for the DiffusionBasedChangeDetector, which integrates multiple change detection methods (Simple Difference, VAE Latent Space, SSIM, and Object Detection).

Input:
Two input images (image1, image2).
Output:
Change map between the two images based on the selected detection method.

Andrew Kiruluta, 08/20/2024
'''
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline
from utils import VAE, LatentDifferenceNetwork, compute_ssim_change_map, filter_change_map_with_objects
from PIL import Image

# Initialize the model
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = 'cpu'


# Step 1: Define the Dataset Class
class ChangeDetectionDataset(Dataset):
    def __init__(self, image_pairs, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1 = Image.open(self.image_pairs[idx][0]).convert('RGB')
        image2 = Image.open(self.image_pairs[idx][1]).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2

# Step 2: Load the Pre-trained Stable Diffusion Model
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Step 3: Define the Feature Extractor Using the Entire VAE
class StableDiffusionFeatureExtractor(nn.Module):
    def __init__(self, vae):
        super(StableDiffusionFeatureExtractor, self).__init__()
        self.vae = vae  # The entire VAE module

    def encode(self, x):
        # The VAE's encode method returns a VAEOutput object with latent_dist
        return self.vae.encode(x).latent_dist.sample()

    def decode(self, z):
        # The VAE's decode method returns a SampleOutput object with 'sample'
        return self.vae.decode(z).sample

# Step 4: Define the Change Detection Model
class DiffusionBasedChangeDetector(nn.Module):
    def __init__(self, feature_extractor):
        super(DiffusionBasedChangeDetector, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, image1, image2):
        z1 = self.feature_extractor.encode(image1)
        z2 = self.feature_extractor.encode(image2)
        z_diff = torch.abs(z1 - z2)
        change_map = self.feature_extractor.decode(z_diff)
        return change_map