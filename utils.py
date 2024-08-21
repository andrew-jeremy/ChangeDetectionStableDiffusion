'''
Description:
This script contains utility functions and additional model components used in the DiffusionBasedChangeDetector, including SSIM computation, VAE architecture, and methods for visualizing change maps.

Input:
Images for computing SSIM, VAE, etc.
Other model components as needed.
Output:
Processed outputs such as SSIM values, latent space differences, and visualized change maps.

Andrew Kiruluta, 08/20/2024
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

# Step 5: Define the Visualization Function
def visualize_change_map(model, image1, image2, title, device='cpu', invert=True):
    model.eval()
    with torch.no_grad():
        image1, image2 = image1.to(device), image2.to(device)
        change_map = model(image1.unsqueeze(0), image2.unsqueeze(0))
        change_map_np = change_map.squeeze().cpu().numpy()
        
        if change_map_np.shape[0] == 3:
            change_map_np = np.mean(change_map_np, axis=0)
        
        if invert:
            change_map_np = 1 - change_map_np
        
        plt.imshow(change_map_np, cmap='gray')
        title = "Generated Change Map" + title
        plt.title(title)
        plt.axis('off')
        plt.show()
        plt.savefig(title + '.png')

# Define the encoder and decoder components of the VAE
class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(256*64*64, latent_dim)
        self.fc_logvar = nn.Linear(256*64*64, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256*64*64)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 64, 64)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

# Define the VAE combining the encoder and decoder for latent difference
"""Variational Autoencoder (VAE) for change detection.
    
    Architecture:
        - Encoder: 
            - 4 convolutional layers with Batch Normalization and ReLU activation.
            - Fully connected layers for learning mean (mu) and log variance (logvar) of the latent space.
        - Decoder:
            - Fully connected layer to reshape the latent vector.
            - 4 transposed convolutional layers with Batch Normalization and ReLU activation.
            - Sigmoid activation at the output layer for reconstruction.
    
    Args:
        input_channels (int): Number of channels in the input images.
        latent_dim (int): Dimensionality of the latent space.
"""
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z


# Define a small neural network to perform non-linear transformation in the latent space
class LatentDifferenceNetwork(nn.Module):
    def __init__(self, latent_dim=128):
        super(LatentDifferenceNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, z1, z2):
        diff = z1 - z2
        x = F.relu(self.fc1(diff))
        x = torch.sigmoid(self.fc2(x))
        return x

# Loss function that includes a learned metric in the latent space
def vae_loss(recon_x, x, mu, logvar, z1, z2, latent_diff_net):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    latent_diff = latent_diff_net(z1, z2)
    return BCE + KLD + latent_diff.mean()

#----------------> Latent Difference VAE Training loop <--------------------------------------------
def train_vae(model, latent_diff_net, dataloader, num_epochs=10, learning_rate=1e-4, device='cuda'):
    optimizer = torch.optim.Adam(list(model.parameters()) + list(latent_diff_net.parameters()), lr=learning_rate)
    model.to(device)
    latent_diff_net.to(device)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            x1, x2 = batch
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()

            # Forward pass
            recon_x1, mu1, logvar1, z1 = model(x1)
            recon_x2, mu2, logvar2, z2 = model(x2)

            # Compute loss
            loss = vae_loss(recon_x1, x1, mu1, logvar1, z1, z2, latent_diff_net)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("VAE Training completed.")

# Alternative SSIM Implementation (No 3D Convolution)
# SSIM calculation and utility functions as previously defined
def gaussian(window_size, sigma):
    gauss = torch.tensor([-(x - window_size // 2)**2 / float(2 * sigma**2) for x in range(window_size)])
    gauss = torch.exp(gauss)
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=False):
    channel = img1.size(1)  # Number of input channels (e.g., 3 for RGB)
    if window is None:
        real_size = min(window_size, img1.size(-1), img1.size(-2))
        window = create_window(real_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

def compute_ssim_change_map(image1, image2, window_size=11):
    """Computes a change map based on the SSIM between two images.
    
    Args:
        image1 (torch.Tensor): The first input image.
        image2 (torch.Tensor): The second input image.
        window_size (int): The size of the Gaussian window for SSIM computation.
        
    Returns:
        torch.Tensor: The SSIM-based change map.
    """
    ssim_map = ssim(image1, image2, window_size=window_size, size_average=False)
    change_map = 1 - ssim_map  # Invert SSIM to get a change likelihood map
    return change_map  # Keep the original dimensions

# Object detection filtering function
def filter_change_map_with_objects(change_map, image1, image2, object_detector, threshold=0.5):
    """
    Filters the change map to keep only the regions where objects are detected.
    
    Args:
        change_map (torch.Tensor): The change map output from the model.
        image1 (torch.Tensor): The first input image.
        image2 (torch.Tensor): The second input image.
        object_detector (torch.nn.Module): The object detection model.
        threshold (float): Confidence threshold for object detection.
        
    Returns:
        torch.Tensor: The filtered change map.
    """
    # Remove the batch dimension and ensure the images are in the right format for object detection
    image1 = image1.squeeze(0)  # Now [3, 512, 512]
    image2 = image2.squeeze(0)  # Now [3, 512, 512]
    image1 = TF.resize(image1, [change_map.size(2), change_map.size(3)])
    image2 = TF.resize(image2, [change_map.size(2), change_map.size(3)])
    
    # Detect objects in both images
    with torch.no_grad():
        detections1 = object_detector([image1])[0]
        detections2 = object_detector([image2])[0]

    # Create a mask for objects detected in both images
    mask1 = torch.zeros_like(change_map, dtype=torch.bool)
    mask2 = torch.zeros_like(change_map, dtype=torch.bool)
    
    for box, score in zip(detections1['boxes'], detections1['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box.int()
            mask1[:, :, y1:y2, x1:x2] = True

    for box, score in zip(detections2['boxes'], detections2['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box.int()
            mask2[:, :, y1:y2, x1:x2] = True

    # Combine the masks and apply them to the change map
    combined_mask = mask1 | mask2
    filtered_change_map = change_map * combined_mask.float()

    return filtered_change_map
