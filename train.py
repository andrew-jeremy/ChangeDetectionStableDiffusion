"""
    A versatile change detection model that combines the power of generative models, 
    perceptual similarity metrics, and object detection. The model can be configured to 
    use one of the following change detection methods:

    1. **Simple Difference Change Map**: Computes a pixel-wise absolute difference 
       between two input images to generate a change map.
    
    2. **VAE Latent Change Map**: Utilizes a Variational Autoencoder (VAE) to encode 
       the input images into a latent space, computes the difference in the latent space, 
       and generates a change map based on the non-linear transformation of this difference.
    
    3. **SSIM-Based Change Map**: Leverages the Structural Similarity Index (SSIM) to 
       create a perceptually-driven change map that focuses on structural differences 
       between the two images.
    
    4. **Object Detection Filtering**: Optionally integrates a pre-trained object 
       detection model to refine the change map by retaining only the regions where 
       identifiable objects are detected.

    The model can be configured to combine these methods, providing flexibility to 
    adapt to different change detection tasks and improve robustness in complex 
    environments.
    
Andrew Kiruluta, 08/18/2024
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from pytorch_msssim import ssim  
import argparse
from torchvision.transforms import functional as TF

from model import DiffusionBasedChangeDetector, ChangeDetectionDataset, StableDiffusionFeatureExtractor, pipeline
from utils import VAE, LatentDifferenceNetwork, compute_ssim_change_map, filter_change_map_with_objects, visualize_change_map, train_vae

# Initialize the model
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = 'cpu'
    
# Step 8: Unsupervised Training Loop
def train_change_detector_unsupervised(model, dataloader, num_epochs=5, learning_rate=1e-4, device=device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for image1, image2 in dataloader:
            image1, image2 = image1.to(device), image2.to(device)

            optimizer.zero_grad()

            # Forward pass
            output_change_map = model(image1, image2)

            # Heuristic-based change map: simple absolute difference
            heuristic_change_map = torch.abs(image1 - image2)

            # Resize heuristic_change_map to match output_change_map dimensions if necessary
            if output_change_map.shape != heuristic_change_map.shape:
                heuristic_change_map = nn.functional.interpolate(heuristic_change_map, size=output_change_map.shape[2:], mode='bilinear', align_corners=False)

            # Loss: minimize difference between model output and heuristic change map
            loss = nn.MSELoss()(output_change_map, heuristic_change_map)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training completed.")

#--> Assume the VAE and LatentDifferenceNetwork classes have already been defined as in the previous implementation
def train_change_detector_vae_unsupervised(model, vae, latent_diff_net, dataloader, num_epochs=10, learning_rate=1e-4, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    vae.to(device)
    latent_diff_net.to(device)
    model.to(device)

    model.train()
    vae.eval()  # VAE is used in evaluation mode to extract latent features

    for epoch in range(num_epochs):
        running_loss = 0.0
        for image1, image2 in dataloader:
            image1, image2 = image1.to(device), image2.to(device)

            optimizer.zero_grad()

            # Forward pass through the VAE to get latent vectors
            with torch.no_grad():
                _, _, _, z1 = vae(image1)
                _, _, _, z2 = vae(image2)

            # Compute the latent space difference using the LatentDifferenceNetwork
            latent_diff = latent_diff_net(z1, z2)

            # Forward pass through the model
            output_change_map = model(image1, image2)

            # Broadcast the latent_diff scalar to the size of the output_change_map
            latent_diff_map = latent_diff.view(-1, 1, 1, 1).expand_as(output_change_map)

            # Compute loss between model output and latent space difference
            loss = nn.MSELoss()(output_change_map, latent_diff_map)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training change_detector model  completed.\n")
#-----------------------------------------------------------------------------------------
# Training loop with dimension matching
def train_change_detector_ssim(model, dataloader, num_epochs=5, learning_rate=1e-4, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for image1, image2 in dataloader:
            image1, image2 = image1.to(device), image2.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            output_change_map = model(image1, image2)

            # Compute the heuristic SSIM-based change map
            heuristic_change_map = compute_ssim_change_map(image1, image2)

            # Resize the heuristic_change_map to match the model's output shape
            heuristic_change_map = F.interpolate(heuristic_change_map, size=output_change_map.shape[2:], mode='bilinear', align_corners=False)

            # Compute loss between model output and heuristic change map
            loss = nn.MSELoss()(output_change_map, heuristic_change_map)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training completed.\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transformer model')
    parser.add_argument('--diff', type=bool, default=True, help='absolute difference')
    parser.add_argument('--vae', type=bool, default=False, help='VAE latent space difference label')
    parser.add_argument('--ssim', type=bool, default=False, help='SSMI space difference label')
    args = parser.parse_args()
    # Fine-tune the model in an unsupervised manner

    # Initialize the VAE and latent difference network
    vae = VAE()
    latent_diff_net = LatentDifferenceNetwork()

    # Step 6: Load Sample Images and Data
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    image_pairs = [
        ("image1.jpg", "image2.jpg"),  # Replace with actual paths to your paired images
    ]

    dataset = ChangeDetectionDataset(image_pairs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Step 7: Initialize the Model
    feature_extractor = StableDiffusionFeatureExtractor(pipeline.vae).to(device)
    change_detector = DiffusionBasedChangeDetector(feature_extractor).to(device)

    if args.diff:
        # train model with simple difference map change map
        title = "simple absolute difference label"
        train_change_detector_unsupervised(change_detector, dataloader, num_epochs=300, learning_rate=1e-4, device=device)
    elif args.vae:
        # Train the latent difference VAE model
        title = "VAE latent space difference label"
        train_vae(vae, latent_diff_net, dataloader, num_epochs=200, learning_rate=1e-6, device=device)
        train_change_detector_vae_unsupervised(change_detector, vae, latent_diff_net, dataloader, num_epochs=200, learning_rate=1e-4, device=device)
    elif args.ssim:
        # uses SSIM calculation 
        title = "SSMI space difference label"
        train_change_detector_ssim(change_detector, dataloader, num_epochs=200, learning_rate=1e-4, device=device)
    
    # Step 9: Test with a pair of images
    image1, image2 = dataset[0]
    visualize_change_map(change_detector, image1, image2, title, device=device, invert=True)
