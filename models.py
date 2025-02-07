import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Initial block
            nn.ConvTranspose2d(z_dim, features_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(True),

            # Reduced number of layers
            *self._make_layer(features_g * 16, features_g * 8),
            *self._make_layer(features_g * 8, features_g * 4),
            *self._make_layer(features_g * 4, features_g * 2),
            *self._make_layer(features_g * 2, features_g),
            
            # Final block
            nn.ConvTranspose2d(features_g, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def _make_layer(self, in_channels, out_channels):
        return [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.2)  # Add dropout for better stability
        ]

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, image_channels, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: image of shape (N, image_channels, 256, 256)
            nn.Conv2d(image_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 8, features_d * 16, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(features_d * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 16, features_d * 32, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(features_d * 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 32, 1, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.net(x)
