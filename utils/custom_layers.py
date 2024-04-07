from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn


class PixelNormLayer(nn.Module):
    """
    This class implements the pixelwise feature vector normalization layer (https://arxiv.org/pdf/1710.10196.pdf)

    Formulation:
    PixelNormLayer(x) = x * (1 / sqrt(mean(x**2) + epsilon)
    """

    def __init__(self, epsilon=1e-8) -> None:
        super().__init__()
        """
        Initializes the PixelNormLayer class
        
        Parameters
        ----------
        epsilon: float
            The epsilon value to use for numerical stability
        """
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the pixel norm layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The normalized input tensor
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class Upscale2d(nn.Module):
    """
    This class implements upscaling layer
    """

    def __init__(self, factor=2, gain=1) -> None:
        super().__init__()
        """
        Initializes the Upscale2d class
        
        Parameters
        ----------
        factor: int
            The upscaling factor
        gain: float
            The gain value to use
        """
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    @staticmethod
    def upscale2d(x: torch.Tensor, factor=2, gain=1):
        """
        Upscales the input tensor by the specified factor

        Parameters
        ----------
        x: torch.Tensor
            The input tensor
        factor: int
            The upscaling factor
        gain: float
            The gain value to use

        Returns
        -------
        torch.Tensor
            The upscaled tensor
        """
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(
                -1, -1, -1, factor, -1, factor
            )
            x = x.contiguous().view(
                shape[0], shape[1], factor * shape[2], factor * shape[3]
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the upscaling layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class Downscale2d(nn.Module):
    """
    This class implements downscaling layer
    """

    def __init__(self, factor=2, gain=1) -> None:
        super().__init__()
        """
        Initializes the Downscale2d class

        Parameters
        ----------
        factor: int
            The downscaling factor
        gain: float
            The gain value to use
        """
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the downscaling layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        return nn.functional.avg_pool2d(x, self.factor)


if __name__ == "__main__":
    pass
