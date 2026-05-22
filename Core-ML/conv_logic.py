import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

class CausalConv1d(nn.Module):
    """
    A 1D Convolution that only looks at the present and the past.
    This is achieved by padding the input by (kernel_size - 1) only on the left.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=0, # We handle padding manually
            dilation=dilation,
            groups=groups
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        # 1. Apply asymmetric padding to the left side only
        # F.pad(input, (left, right))
        x = F.pad(x, (self.padding, 0))
        
        # 2. Run the standard convolution
        return self.conv(x)

class DepthwiseSeparableCausalConv1d(nn.Module):
    """
    A more efficient version of CausalConv1d.
    It splits the convolution into two steps:
    1. Depthwise (Spatial patterns across words)
    2. Pointwise (Mixing channel information)
    """
    def __init__(self, dim, kernel_size, dilation=1):
        super().__init__()
        # Depthwise: One kernel per channel (groups=dim)
        self.depthwise = CausalConv1d(
            dim, dim, kernel_size, dilation=dilation, groups=dim   #If this isn't Casual, PyTorch would REFUSE to do aysmmetric padding
        )
        # Pointwise: 1x1 convolution to mix information
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Dim, Length)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
