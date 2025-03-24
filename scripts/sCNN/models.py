
import math
import torch
import torch.nn as nn
import healpy as hp
import numpy as np

from .sh import spherical_harmonic
from .layers import SphConv
from .sh import ls, sft, isft

import torch.nn.functional as F

class MLPModel(torch.nn.Module):
    """A simple multi-layer perceptron model."""

    def __init__(self, n_in: int, n_out: int):
        """Initialize the model layers.

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output units.
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(n_in, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass.

        Args:
            x: Input signals.

        Returns:
            Signals after passing through the network.
        """

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        return x
        

class SCNNModel_3shells_with_attention(torch.nn.Module):
    def __init__(self, c_in: int, n_out: int):
        super().__init__()
        self.register_buffer("ls", ls)
        self.register_buffer("sft", sft)
        self.register_buffer("isft", isft)

        # Shell-specific processing
        self.conv_shell1 = SphConv(1, 16)
        self.conv_shell2 = SphConv(1, 16)
        self.conv_shell3 = SphConv(1, 16)

        # Attention-based fusion
        self.shell_attention = ShellAttention()

        # Encoder
        self.conv2 = SphConv(48, 32)
        self.conv3 = SphConv(32, 64)
        self.conv3a = SphConv(64, 64)
        # Decoder
        self.conv4 = SphConv(64, 32)
        self.conv4a = SphConv(32, 32)
        self.conv5 = SphConv(32, 16)
        self.conv6 = SphConv(16, 1)

        # FC layers
        self.fc1 = torch.nn.Linear(128, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, n_out)

    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.sft
            @ torch.nn.functional.leaky_relu(
                self.isft @ x.unsqueeze(-1),
                negative_slope=0.1,
            )
        ).squeeze(-1)

    def global_pooling(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.isft @ x.unsqueeze(-1), dim=2).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process shells with attention fusion
        x_shell1 = self.conv_shell1(x[:, 0:1, :])
        x_shell2 = self.conv_shell2(x[:, 1:2, :])
        x_shell3 = self.conv_shell3(x[:, 2:3, :])
        
        # Attention-based fusion instead of simple concatenation
        x = self.shell_attention([x_shell1, x_shell2, x_shell3])

        # Deeper encoder
        x = self.conv2(x)
        x1 = self.nonlinearity(x)
        x = self.conv3(x1)
        x = self.conv3a(x)  
        x2 = self.nonlinearity(x)

        # Deeper decoder
        x = self.conv4(x2)
        x = self.conv4a(x) 
        x3 = self.nonlinearity(x)
        x = self.conv5(x3)
        x = self.nonlinearity(x)

        odfs_sh = self.conv6(x).squeeze(1)[:, 0:45]

        # FC network
        x = torch.cat([self.global_pooling(x1), 
                      self.global_pooling(x2),
                      self.global_pooling(x3)], dim=1)
        
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        odfs_sh += self.fc3(x)

        return odfs_sh


class ShellAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(48, 24),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(24, 3),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, shells):
        # shells: list of [batch, 16, coefficients]
        concat = torch.cat([s.mean(-1) for s in shells], dim=-1)
        weights = self.attention_net(concat)
        
        # Apply attention weights
        weighted = []
        for i, shell in enumerate(shells):
            weighted.append(shell * weights[:, i].view(-1, 1, 1))
            
        return torch.cat(weighted, dim=1)





# class SCNNModel_3shells_optimized(nn.Module):
#     def __init__(self, c_in: int, n_out: int, l_max: int = 8, n_sides: int = 16):
#         super().__init__()
#         self.l_max = l_max
        
#         # Initialize spherical harmonic transforms
#         self._init_sh_transforms(n_sides)
        
#         # Shell-specific processing
#         self.conv_shell1 = SphConv(1, 16, l_max)
#         self.bn_shell1 = nn.BatchNorm1d(16)
#         self.conv_shell2 = SphConv(1, 16, l_max)
#         self.bn_shell2 = nn.BatchNorm1d(16)
#         self.conv_shell3 = SphConv(1, 16, l_max)
#         self.bn_shell3 = nn.BatchNorm1d(16)

#         # Attention-based fusion
#         self.shell_attention = ShellAttention(channels=16, num_shells=3)

#         # Encoder-Decoder with skip connections
#         self.encoder = nn.ModuleList([
#             SphConvBlock(48, 32, l_max),
#             SphConvBlock(32, 64, l_max),
#             SphConvBlock(64, 64, l_max)
#         ])
        
#         self.decoder = nn.ModuleList([
#             SphConvBlock(64, 32, l_max),
#             SphConvBlock(32, 32, l_max),
#             SphConvBlock(32, 16, l_max)
#         ])

#         # Final projection
#         self.final_conv = SphConv(16, 1, l_max)
        
#         # FC layers with dropout
#         self.fc = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),
#             nn.Linear(128, n_out)
#         )

#         # Initialize weights
#         self._init_weights()

#     def _init_sh_transforms(self, n_sides):
#         # Generate HEALPix grid and SH matrices
#         vertices = torch.tensor(hp.pix2vec(n_sides, np.arange(12 * n_sides**2))).T
#         thetas = torch.arccos(vertices[:, 2])
#         phis = (torch.arctan2(vertices[:, 1], vertices[:, 0]) + 2 * np.pi) % (2 * np.pi)
        
#         # Create SFT and ISFT matrices
#         n_coeffs = (self.l_max + 1) * (self.l_max + 2) // 2
#         isft = torch.zeros((len(vertices), n_coeffs))
        
#         for l in range(0, self.l_max+1, 2):
#             for m in range(-l, l+1):
#                 idx = l*(l+1)//2 + m
#                 isft[:, idx] = spherical_harmonic(l, m, thetas, phis)
        
#         sft = torch.linalg.pinv(isft.T @ isft) @ isft.T
        
#         # Register buffers
#         self.register_buffer('sft', sft)
#         self.register_buffer('isft', isft)
#         self.register_buffer('vertices', vertices)

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def spherical_nonlinearity(self, x):
#         """Apply SFT -> LeakyReLU -> ISFT"""
#         x_sft = torch.einsum('vj,bcj->bvc', self.sft, x)
#         x_act = F.leaky_relu(x_sft, 0.1)

#         x_isft = torch.einsum('jv,bvc->bcj', self.isft, x_act)
        
#         return x_isft

#     def forward(self, x):
#         # Shell processing
#         x1 = self.spherical_nonlinearity(self.bn_shell1(self.conv_shell1(x[:, 0:1])))
#         x2 = self.spherical_nonlinearity(self.bn_shell2(self.conv_shell2(x[:, 1:2])))
#         x3 = self.spherical_nonlinearity(self.bn_shell3(self.conv_shell3(x[:, 2:3])))
        
#         # Attention fusion
#         x = self.shell_attention([x1, x2, x3])

#         # Encoder
#         skip_connections = []
#         for layer in self.encoder:
#             x = layer(x)
#             skip_connections.append(x)

#         # Decoder
#         for i, layer in enumerate(self.decoder):
#             x = layer(x) + skip_connections[-i-2]  # Skip connections

#         # Final projection
#         odfs_sh = self.final_conv(x).squeeze(1)[:, :45]

#         # Global pooling and FC
#         pooled = torch.cat([self.isft @ x.mean(dim=-1) for x in skip_connections], dim=1)
#         fc_out = self.fc(pooled)
        
#         return odfs_sh + fc_out  # Residual connection

# class SphConvBlock(nn.Module):
#     """Basic building block with Conv-BN-Activation"""
#     def __init__(self, c_in, c_out, l_max):
#         super().__init__()
#         self.conv = SphConv(c_in, c_out, l_max)
#         self.bn = nn.BatchNorm1d(c_out)
        
#     def forward(self, x):
#         return F.leaky_relu(self.bn(self.conv(x)), 0.1)

# class ShellAttention(nn.Module):
#     def __init__(self, channels=16, num_shells=3):
#         super().__init__()
#         self.channels = channels
#         self.num_shells = num_shells
        
#         # Context networks
#         self.context_nets = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(channels, 32, 1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(32, 16, 1)
#             ) for _ in range(num_shells)
#         ])
        
#         # Attention network
#         self.attention = nn.Sequential(
#             nn.Linear(16 * num_shells, 32),
#             nn.LeakyReLU(0.1),
#             nn.Linear(32, num_shells),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, shells):
#         contexts = []
#         for shell, net in zip(shells, self.context_nets):
#             contexts.append(net(shell).mean(dim=-1))  # [B, 16]
            
#         weights = self.attention(torch.cat(contexts, dim=-1))  # [B, 3]
#         return sum(w.unsqueeze(1).unsqueeze(-1) * shell 
#                  for w, shell in zip(weights.unbind(-1), shells))

# class SphConv(nn.Module):
#     def __init__(self, c_in: int, c_out: int, l_max: int):
#         super().__init__()
#         self.l_max = l_max
#         self.register_buffer("expansion_mask", self._create_expansion_mask())
        
#         # Learnable parameters
#         self.weights = nn.Parameter(torch.empty(c_out, c_in, (l_max//2)+1))
#         self.scale = nn.Parameter(torch.ones(1))
        
#         # Residual connection
#         self.residual = c_in == c_out
#         if not self.residual:
#             self.res_proj = nn.Conv1d(c_in, c_out, 1)

#         nn.init.kaiming_uniform_(self.weights, a=0.1)

#     def _create_expansion_mask(self):
#         mask = []
#         for l in range(0, self.l_max+1, 2):
#             mask += [l//2] * (2*l + 1)
#         return torch.tensor(mask, dtype=torch.long)

#     def forward(self, x):
#         expanded_weights = self.weights[:, :, self.expansion_mask]
#         out = torch.einsum('bci,oci->boi', x, expanded_weights) * self.scale
        
#         if self.residual:
#             return out + x
#         return out + self.res_proj(x) if hasattr(self, 'res_proj') else out
