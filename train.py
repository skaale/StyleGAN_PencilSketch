import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PencilSketchDataset
from models import Generator, Discriminator

# --- Hyperparameters ---
BATCH_SIZE = 8
IMAGE_SIZE = 256
Z_DIM = 100
LR = 1e-4
EPOCHS = 1000
FEATURES_G = 32
FEATURES_D = 32
IMAGE_CHANNELS = 3
CRITIC_ITERATIONS = 3
LAMBDA_GP = 10

# Directory setup - only checkpoints folder
CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Dataset
dataset = PencilSketchDataset(root_dir="downloaded_images", image_size=IMAGE_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print("Total images in dataset:", len(dataset))

# Initialize models
generator = Generator(z_dim=Z_DIM, image_channels=IMAGE_CHANNELS, features_g=FEATURES_G).to(device)
discriminator = Discriminator(image_channels=IMAGE_CHANNELS, features_d=FEATURES_D).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# Gradient penalty function
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# After model initialization
print(f"Generator device: {next(generator.parameters()).device}")
print(f"Discriminator device: {next(discriminator.parameters()).device}")

# Training Loop
for epoch in range(EPOCHS):
    for batch_idx, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        if batch_idx == 0:  # Print only for first batch
            print(f"Batch device: {real_images.device}")
        batch_size = real_images.size(0)
        
        # Train Discriminator
        for _ in range(CRITIC_ITERATIONS):
            optimizer_D.zero_grad()
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            real_score = discriminator(real_images)
            fake_score = discriminator(fake_images.detach())
            gp = compute_gradient_penalty(discriminator, real_images, fake_images.detach())
            loss_D = torch.mean(fake_score) - torch.mean(real_score) + LAMBDA_GP * gp
            loss_D.backward()
            optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_score = discriminator(fake_images)
        loss_G = -torch.mean(fake_score)
        loss_G.backward()
        optimizer_G.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Save only the latest checkpoint, overwriting the previous one
    checkpoint = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "generator_checkpoint.pth"))

print("Training complete!")