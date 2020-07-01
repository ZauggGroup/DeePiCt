import torch
import torch.nn as nn

import src.python.tensors.actions as actions


class UNet(nn.Module):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    # conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 3d convolutions with elu activation
    def _conv_block_large_padding(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU())

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU())

    def _conv_block_dropout(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=0.4),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=0.4))

    # upsampling via transposed 3d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels,
                                  kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1,
                 final_activation=None):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = 4

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation,
                              nn.Module), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters)
        #  must be wraped itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList([self._conv_block(in_channels, 2),
                                      self._conv_block(2, 4),
                                      self._conv_block(4, 16),
                                      self._conv_block(16, 32)])
        # the base convolution block
        self.base = self._conv_block(32, 32)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(32, 16),
                                      self._conv_block(16, 8),
                                      self._conv_block(8, 4),
                                      self._conv_block(4, 2)])

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList(
            [nn.MaxPool3d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(32, 16),
                                         self._upsampler(16, 8),
                                         self._upsampler(8, 4),
                                         self._upsampler(4, 2)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv3d(2, out_channels, 1)
        self.activation = final_activation

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = actions.crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](
                self._crop_and_concat(x, encoder_out[level]))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet_deep(UNet):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    # conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 3d convolutions with elu activation
    def _conv_block_deep(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU()
        )


    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels,
                                  kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1,
                 final_activation=None):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = 4

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation,
                              nn.Module), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters)
        #  must be wraped itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList([self._conv_block_deep(in_channels, 16),
                                      self._conv_block(16, 64),
                                      self._conv_block(64, 128),
                                      self._conv_block(128, 256)])
        # the base convolution block
        self.base = self._conv_block(256, 256)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(256, 128),
                                      self._conv_block(128, 64),
                                      self._conv_block(64, 32),
                                      self._conv_block(32, 16)])

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList(
            [nn.MaxPool3d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(256, 128),
                                         self._upsampler(128, 64),
                                         self._upsampler(64, 32),
                                         self._upsampler(32, 16)])
        self.out_conv = nn.Conv3d(16, out_channels, 1)


class UNet_7(UNet):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    def __init__(self, in_channels=1, out_channels=1,
                 final_activation=None):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to:
        self.depth = 7

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation,
                              nn.Module), "Activation must be torch module"

        # all lists of convolution layers (or other nn.Modules with parameters)
        # must be wrapped into a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList([self._conv_block(in_channels, 2),
                                      self._conv_block(2, 4),
                                      self._conv_block(4, 16),
                                      self._conv_block(16, 32),
                                      self._conv_block(32, 64),
                                      self._conv_block(64, 128),
                                      self._conv_block(128, 256)])
        self.base = self._conv_block(256, 256)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(256, 128),
                                      self._conv_block(128, 64),
                                      self._conv_block(64, 32),
                                      self._conv_block(32, 16),
                                      self._conv_block(16, 8),
                                      self._conv_block(8, 4),
                                      self._conv_block(4, 2)])
        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList(
            [nn.MaxPool3d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(256, 128),
                                         self._upsampler(128, 64),
                                         self._upsampler(64, 32),
                                         self._upsampler(32, 16),
                                         self._upsampler(16, 8),
                                         self._upsampler(8, 4),
                                         self._upsampler(4, 2)])
        # output convolution and activation
        self.out_conv = nn.Conv3d(2, out_channels, 1)
        self.activation = final_activation


class UNet_4(UNet):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    def __init__(self, in_channels=1, out_channels=1,
                 final_activation=None):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to:
        self.depth = 4

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation,
                              nn.Module), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList([self._conv_block(in_channels, 2),
                                      self._conv_block(2, 4),
                                      self._conv_block(4, 16),
                                      self._conv_block(16, 32)])
        self.base = self._conv_block(32, 32)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(32, 16),
                                      self._conv_block(16, 8),
                                      self._conv_block(8, 4),
                                      self._conv_block(4, 2)])
        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList(
            [nn.MaxPool3d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(32, 16),
                                         self._upsampler(16, 8),
                                         self._upsampler(8, 4),
                                         self._upsampler(4, 2)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv3d(2, out_channels, 1)
        self.activation = final_activation


class UNet_6(UNet):
    def __init__(self, in_channels=1, out_channels=1,
                 final_activation=None):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to:
        self.depth = 6

        self.encoder = nn.ModuleList([self._conv_block(in_channels, 2),
                                      self._conv_block(2, 4),
                                      self._conv_block(4, 16),
                                      self._conv_block(16, 32),
                                      self._conv_block(32, 64),
                                      self._conv_block(64, 128)])

        # self.base = self.conv_block(512, 512)
        self.base = self._conv_block(128, 128)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(128, 64),
                                      self._conv_block(64, 32),
                                      self._conv_block(32, 16),
                                      self._conv_block(16, 8),
                                      self._conv_block(8, 4),
                                      self._conv_block(4, 2)])

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList(
            [nn.MaxPool3d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(128, 64),
                                         self._upsampler(64, 32),
                                         self._upsampler(32, 16),
                                         self._upsampler(16, 8),
                                         self._upsampler(8, 4),
                                         self._upsampler(4, 2)])

        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv3d(2, out_channels, 1)
        self.activation = final_activation
