# ===============================================
# DEEP CONVOLUTIONAL GAN (DCGAN)
# Generates complex color images (synthetic shapes)
# Fixed Discriminator + Loss Curve Visualization
# ===============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

# ----------------------------
# PARAMETERS
# ----------------------------
image_size = 64
batch_size = 64
nz = 100        # latent vector size
ngf = 128       # generator filters
ndf = 128       # discriminator filters
epochs = 5
lr = 0.0002
beta1 = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# SYNTHETIC SHAPES DATASET
# ----------------------------
class ShapesDataset(Dataset):
    def __init__(self, length=10000, size=64):
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = Image.new('RGB', (self.size, self.size), (random.randint(0,255),)*3)
        draw = ImageDraw.Draw(img)

        for _ in range(random.randint(1,5)):
            shape = random.choice(['rectangle', 'ellipse'])
            x1, y1 = random.randint(0, self.size-20), random.randint(0, self.size-20)
            x2, y2 = x1 + random.randint(10, self.size-x1), y1 + random.randint(10, self.size-y1)
            # Ensure valid coordinates
            if x2 > x1 and y2 > y1:
                color = tuple(np.random.randint(0,255,3).tolist())
                if shape == 'rectangle':
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    draw.ellipse([x1, y1, x2, y2], fill=color)

        img = np.array(img).transpose(2,0,1)
        img = torch.FloatTensor(img) / 127.5 - 1
        return img

dataset = ShapesDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Batches per epoch:", len(dataloader))

# ----------------------------
# GENERATOR
# ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# ----------------------------
# FIXED DISCRIMINATOR
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.main(input)
        out = torch.mean(out, dim=[2, 3])  # global average pooling (N,1)
        return self.sigmoid(out).view(-1)

# ----------------------------
# INITIALIZATION
# ----------------------------
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

G_losses, D_losses = [], []

# ----------------------------
# TRAINING LOOP
# ----------------------------
print("Starting training...")
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        real = data.to(device)
        b_size = real.size(0)
        label_real = torch.full((b_size,), 1., device=device)
        label_fake = torch.full((b_size,), 0., device=device)

        # --- Update D ---
        netD.zero_grad()
        output = netD(real)
        errD_real = criterion(output, label_real)
        errD_real.backward()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, label_fake)
        errD_fake.backward()
        optimizerD.step()

        # --- Update G ---
        netG.zero_grad()
        output = netD(fake)
        errG = criterion(output, label_real)
        errG.backward()
        optimizerG.step()

        # Save losses for plotting
        G_losses.append(errG.item())
        D_losses.append((errD_real + errD_fake).item())

        if i % 100 == 0:
            print(f"[{epoch+1}/{epochs}] [{i}/{len(dataloader)}] Loss_D: {(errD_real+errD_fake).item():.3f} Loss_G: {errG.item():.3f}")

    # --- Show generated images ---
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    grid = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"Generated Images - Epoch {epoch+1}")
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()

# ----------------------------
# PLOT LOSS CURVES
# ----------------------------
plt.figure(figsize=(8,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G Loss")
plt.plot(D_losses, label="D Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
