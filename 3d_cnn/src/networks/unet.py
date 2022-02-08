import torch
import torch.nn as nn

import tensors.actions as actions


class UNet3D(nn.Module):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
      depth: depth of the u-net (= number of encoder / decoder levels)
      initial_features: number of features after first encoder
    """

    def _conv_block_BN_encoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.encoder_dropout)
        )

    def _conv_block_BN_encoder_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ELU(),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ELU(),
            nn.Dropout(p=self.encoder_dropout))

    def _conv_block_BN_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.decoder_dropout)
        )

    def _conv_block_decoder_BN_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ELU(),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ELU(),
            nn.Dropout(p=self.decoder_dropout))

    def _conv_block_encoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=self.encoder_dropout))

    def _conv_block_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=self.decoder_dropout))

    def _conv_block_encoder_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.encoder_dropout))

    def _conv_block_decoder_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.decoder_dropout))

    # upsampling via transposed 3d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels,
                                  kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1,
                 depth=4, initial_features=16, decoder_dropout=0.1,
                 encoder_dropout=0.1, BN: bool = False, elu=False,
                 final_activation=None):
        super().__init__()
        self.depth = depth
        self.decoder_dropout = decoder_dropout
        self.encoder_dropout = encoder_dropout
        self.non_linearity = nn.ELU() if elu else nn.ReLU
        self.batch_norm = BN

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation,
                              nn.Module), "Activation must be torch module"

        n_features = [initial_features * 2 ** level
                      for level in range(self.depth)]
        # modules of the encoder path
        n_features_encode = [in_channels] + n_features
        n_features_base = n_features_encode[-1] * 2

        if not elu:
            if BN:
                self.encoder = nn.ModuleList(
                    [self._conv_block_BN_encoder(n_features_encode[level],
                                                 n_features_encode[level + 1])
                     for level in range(self.depth)])

                # the base convolution block
                self.base = self._conv_block_encoder(n_features_encode[-1],
                                                     n_features_base)
            else:
                self.encoder = nn.ModuleList(
                    [self._conv_block_encoder(n_features_encode[level],
                                              n_features_encode[level + 1])
                     for level in range(self.depth)])

                # the base convolution block
                self.base = self._conv_block_encoder(n_features_encode[-1],
                                                     n_features_base)
        else:
            if BN:
                self.encoder = nn.ModuleList(
                    [self._conv_block_BN_encoder_elu(n_features_encode[level],
                                                     n_features_encode[
                                                         level + 1])
                     for level in range(self.depth)])

                # the base convolution block
                self.base = self._conv_block_encoder_elu(n_features_encode[-1],
                                                         n_features_base)
            else:
                self.encoder = nn.ModuleList(
                    [self._conv_block_encoder_elu(n_features_encode[level],
                                                  n_features_encode[
                                                      level + 1])
                     for level in range(self.depth)])

                # the base convolution block
                self.base = self._conv_block_encoder_elu(n_features_encode[-1],
                                                         n_features_base)
        # modules of the decoder path
        n_features_decode = [n_features_base] + n_features[::-1]
        # print("decoder:", n_features_decode)
        if not elu:
            if BN:
                self.decoder = nn.ModuleList(
                    [self._conv_block_BN_decoder(n_features_decode[level],
                                                 n_features_decode[
                                                     level + 1])
                     for level in range(self.depth)])
            else:
                self.decoder = nn.ModuleList(
                    [self._conv_block_decoder(n_features_decode[level],
                                              n_features_decode[
                                                  level + 1])
                     for level in range(self.depth)])

        else:
            if BN:
                self.decoder = nn.ModuleList(
                    [self._conv_block_decoder_BN_elu(n_features_decode[level],
                                                     n_features_decode[
                                                         level + 1])
                     for level in range(self.depth)])
            else:
                self.decoder = nn.ModuleList(
                    [self._conv_block_decoder_elu(n_features_decode[level],
                                                  n_features_decode[
                                                      level + 1])
                     for level in range(self.depth)])

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList(
            [nn.MaxPool3d(2) for _ in range(self.depth)])

        # the upsampling layers
        self.upsamplers = nn.ModuleList(
            [self._upsampler(n_features_decode[level],
                             n_features_decode[level + 1])
             for level in range(self.depth)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv3d(initial_features, out_channels, 1)
        self.activation = final_activation

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = actions.crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward(self, input_tensor):
        x = input_tensor
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
