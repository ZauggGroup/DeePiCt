import torch
import torch.nn as nn

from networks.layers import GroupNorm3d


# 3D convolution layer with batch normalization and ReLu activation
# (from https://github.com/JorisRoels/domain-adaptive-segmentation)
class ConvBatchNormRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1):
        super(ConvBatchNormRelu3D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else:  # VALID (no) padding
            p = 0
        self.unit = nn.Sequential(nn.Conv3d(int(in_channels), int(out_channels),
                                            kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias,
                                            dilation=dilation),
                                  nn.BatchNorm3d(int(out_channels)),
                                  nn.ReLU(), )

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs


class ConvGroupNormRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1):
        super(ConvGroupNormRelu3D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else:  # VALID (no) padding
            p = 0
        self.unit = nn.Sequential(nn.Conv3d(int(in_channels), int(out_channels),
                                            kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias,
                                            dilation=dilation),
                                  GroupNorm3d(int(out_channels)),
                                  nn.ReLU(), )

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs


class UNetConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME',
                 group_norm=True):
        super(UNetConvBlock3D, self).__init__()

        if group_norm:
            self.conv1 = ConvGroupNormRelu3D(in_channels, out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding)
            self.conv2 = ConvGroupNormRelu3D(out_channels, out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding)
        else:
            self.conv1 = ConvBatchNormRelu3D(in_channels, out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding)
            self.conv2 = ConvBatchNormRelu3D(out_channels, out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUpSamplingBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock3D, self).__init__()

        if deconv:  # use transposed convolution
            self.up = nn.ConvTranspose3d(in_channels, out_channels,
                                         kernel_size=2, stride=2, bias=bias)
        else:  # use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, *arg):
        if len(arg) == 2:
            return self.forward_concat(arg[0], arg[1])
        else:
            return self.forward_standard(arg[0])

    def forward_concat(self, inputs1, inputs2):

        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):

        return self.up(inputs)


class conv_block3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1, non_linearity_module: nn.Module = nn.ReLU()):
        super(conv_block3D, self).__init__()
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.non_linearity_module = non_linearity_module

        self.unit = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size, padding=self.padding),
            self.non_linearity_module,
            nn.Conv3d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      padding=self.kernel_size),
            self.non_linearity_module)


def forward(self, inputs):
    outputs = self.unit(inputs)
    return outputs
