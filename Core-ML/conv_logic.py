import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

class CausalConv1d(nn.Module):
    """
    A 1D Convolutional layer restricted to look only at current and past tokens.
    Achieved by prepending padding of size (kernel_size - 1) * dilation to the sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Calculate padding needed on the left (scalar integer)
        self.padding = (kernel_size - 1) * dilation
        
        # Instantiate standard PyTorch Conv1d
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=0, # Set padding to 0 because we perform manual asymmetric left-padding
            dilation=dilation,
            groups=groups
        )

    def forward(self, x):
        # Input shape x: [Batch, Channels, Length]
        # F.pad padding tuple layout: (left, right) padding sizes
        # Pads the left side of the sequence by `self.padding` elements, preserving causality
        # Padded tensor shape: [Batch, Channels, Length + padding]
        x = F.pad(x, (self.padding, 0))
        
        # Convolve padded sequence; output shape: [Batch, Channels, Length]
        return self.conv(x)

class DepthwiseSeparableCausalConv1d(nn.Module):
    """
    Depthwise-Separable 1D Causal Convolution.
    Decouples spatial filtering (depthwise) from channel-wise mixing (pointwise) for parameter efficiency.
    """
    def __init__(self, dim, kernel_size, dilation=1):
        super().__init__()
        # Depthwise step: Apply spatial causal convolution independently per channel (groups=dim)
        self.depthwise = CausalConv1d(
            dim, dim, kernel_size, dilation=dilation, groups=dim
        )
        # Pointwise step: 1x1 convolution to mix representations across channels
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        # Input shape x: [Batch, Dim, Length]
        # Apply depthwise spatial filtering; output shape: [Batch, Dim, Length]
        x = self.depthwise(x)
        # Apply pointwise channel mixing; output shape: [Batch, Dim, Length]
        x = self.pointwise(x)
        return x
