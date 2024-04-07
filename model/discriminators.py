import argparse
from typing import List, Tuple
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from model.networks import BaseNetwork
from utils.custom_layers import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ACGANDiscriminator(BaseNetwork):
    """
    This class implements the ACGANDiscriminator
    """

    def __init__(
        self, opt: argparse.Namespace, n_classes: int, in_channels: int
    ) -> None:
        """
        Initializes the ACGANDiscriminator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        in_channels: int
            The number of input channels
        """
        super().__init__()
        self._name = "ACGANDiscriminator"
        self._opt = opt

        self.conv_blocks = nn.Sequential(
            *self._discriminator_block(in_channels, 16, bn=False),
            *self._discriminator_block(16, 32),
            *self._discriminator_block(32, 64),
            *self._discriminator_block(64, 128),
        )

        # Dimension of output feature map after conv_blocks
        self.dim = self._opt.img_size // 2**4

        self.adv_layer = nn.Linear(128 * self.dim**2, 1)
        self.aux_layer = nn.Linear(128 * self.dim**2, n_classes)

    def _discriminator_block(
        self, in_filters: int, out_filters: int, bn: bool = True
    ) -> List:
        """
        Creates a discriminator block

        Parameters
        ----------
        in_filters: int
            The number of input filters
        out_filters: int
            The number of output filters
        bn: bool
            Whether to use batch normalization or not

        Returns
        -------
        list
            The discriminator block
        """
        block = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        ]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        img: torch.Tensor
            The input image

        Returns
        -------
        validity: torch.Tensor
            The validity of the image (real or fake)
        label: torch.Tensor
            The label of the image
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        validity = torch.sigmoid(validity)
        label = self.aux_layer(out)
        # label = torch.softmax(label, dim=1)
        return validity, label


class VanillaDiscriminator(BaseNetwork):
    """
    This class implements the VanillaDiscriminator
    """

    def __init__(
        self, opt: argparse.Namespace, d_neurons: int, out_channels: int
    ) -> None:
        """
        Initializes the VanillaDiscriminator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        d_neurons: int
            The number of neurons in the discriminator
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "VanillaDiscriminator"
        self._opt = opt
        self.in_size = self._opt.img_size * self._opt.img_size * out_channels

        self.linear1 = nn.Linear(self.in_size, d_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        torch.Tensor
            The validity of the image (real or fake)
        """
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear4(out)
        validity = torch.sigmoid(out)
        return validity


class ACVanillaDiscriminator(BaseNetwork):
    """
    This class implements the ACVanillaDiscriminator
    """

    def __init__(
        self, opt: argparse.Namespace, n_classes: int, d_neurons: int, out_channels: int
    ) -> None:
        """
        Initializes the ACVanillaDiscriminator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        d_neurons: int
            The number of neurons in the discriminator
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACVanillaDiscriminator"
        self._opt = opt
        self.in_size = self._opt.img_size * self._opt.img_size * out_channels
        self.n_classes = n_classes

        self.linear1 = nn.Linear(self.in_size, d_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )
        self.adv_layer = nn.Linear(self.linear3.out_features, 1)
        self.aux_layer = nn.Linear(self.linear3.out_features, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward pass for the discriminator

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        validity: torch.Tensor
            The validity of the image (real or fake)
        label: torch.Tensor
            The label of the image
        """
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        validity = self.adv_layer(out)
        validity = torch.sigmoid(validity)
        label = self.aux_layer(out)
        return validity, label


class WGANCritic(BaseNetwork):
    """
    This class implements the WGANCritic
    """

    def __init__(
        self, opt: argparse.Namespace, d_neurons: int, out_channels: int
    ) -> None:
        """
        Initializes the WGANCritic class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        d_neurons: int
            The number of neurons in the critic
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "WGANCritic"
        self._opt = opt
        self.in_size = self._opt.img_size * self._opt.img_size * out_channels

        self.linear1 = nn.Linear(self.in_size, d_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the critic

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        torch.Tensor
            The realness of the image
        """
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear4(out)
        validity = out
        return validity


class ConvGANCritic(BaseNetwork):
    """
    This class implements the ConvGANCritic
    """

    def __init__(self, opt: argparse.Namespace, in_channels: int) -> None:
        """
        Initializes the ConvGANCritic class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        in_channels: int
            The number of input channels
        """
        super().__init__()
        self._name = "ConvGANCritic"
        self._opt = opt

        self.conv_blocks = nn.Sequential(
            *self._discriminator_block(in_channels, 16, bn=False),
            *self._discriminator_block(16, 32),
            *self._discriminator_block(32, 64),
            *self._discriminator_block(64, 128),
        )

        # Dimension of output feature map after conv_blocks
        self.dim = self._opt.img_size // 2**4

        self.real_layer = nn.Linear(128 * self.dim**2, 1)

    def _discriminator_block(
        self, in_filters: int, out_filters: int, bn: bool = True
    ) -> List:
        """
        Creates a discriminator block

        Parameters
        ----------
        in_filters: int
            The number of input filters
        out_filters: int
            The number of output filters
        bn: bool
            Whether to use batch normalization or not

        Returns
        -------
        list
            The discriminator block
        """
        block = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        ]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        img: torch.Tensor
            The input image

        Returns
        -------
        torch.Tensor
            The realness of the image
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.real_layer(out)
        return validity

