import argparse
from collections import OrderedDict
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
from model.networks import BaseNetwork
from utils.custom_layers import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ACGANGenerator(BaseNetwork):
    """
    This class implements the ACGAN generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        n_classes: int,
        embedding_dim: int,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the ACGANGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        embedding_dim: int
            The embedding dimension of the labels
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACGANGenerator"
        self._opt = opt
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels

        self.label_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.init_size = self._opt.img_size // 4
        # self.init_linear = nn.Linear(self.embedding_dim+self.latent_dim, 128*self.init_size**2)
        self.init_linear = nn.Linear(self.latent_dim, 128 * self.init_size**2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor
        labels: torch.Tensor
            The labels tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        # gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        gen_input = torch.mul(self.label_embedding(labels), noise)
        out = self.init_linear(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ConvGANGenerator(BaseNetwork):
    """
    This class implements the ConvGAN generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the ACGANGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ConvGANGenerator"
        self._opt = opt
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.init_size = self._opt.img_size // 4
        self.init_linear = nn.Linear(self.latent_dim, 128 * self.init_size**2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        out = self.init_linear(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class VanillaGenerator(BaseNetwork):
    """
    This class implements the vanilla generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        g_neurons: int,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the VanillaGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        g_neurons: int
            The number of neurons in the generator
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "VanillaGenerator"
        self._opt = opt
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.out_size = self._opt.img_size * self._opt.img_size * self.out_channels

        self.linear1 = nn.Linear(self.latent_dim, g_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features * 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features * 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, self.out_size)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        out = self.linear1(noise)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear4(out)
        out = nn.Tanh()(out)
        return out


class ACVanillaGenerator(BaseNetwork):
    """
    This class implements the ACVanillaGAN generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        n_classes: int,
        embedding_dim: int,
        g_neurons: int,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the ACVanillaGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        embedding_dim: int
            The embedding dimension of the labels
        g_neurons: int
            The number of neurons in the generator
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACVanillaGenerator"
        self._opt = opt
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.out_size = self._opt.img_size * self._opt.img_size * self.out_channels

        self.label_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.linear1 = nn.Linear(self.latent_dim, g_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features * 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features * 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, self.out_size)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor
        labels: torch.Tensor
            The labels tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        gen_input = torch.mul(self.label_embedding(labels), noise)
        out = self.linear1(gen_input)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear4(out)
        out = nn.Tanh()(out)
        return out


