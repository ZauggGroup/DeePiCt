import os
from os.path import join

import numpy as np
import tensorboardX as tb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import tensors.actions as actions
from networks.metrics import jaccard, accuracy_metrics


# Implementation inspired from:
# https://github.com/JorisRoels/domain-adaptive-segmentation
# that separates encoder and decoder paths


class UNetEncoder3D(nn.Module):
    def conv_block(self, in_channels, out_channels):
        # I have personally had better experience using relu instead of elu,
        # but this is worth confirming experimentally
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU())

    def conv_block_elu(self, in_channels, out_channels):
        # I have personally had better experience using relu instead of elu,
        # but this is worth confirming experimentally
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU())

    def __init__(self, in_channels=1, depth=4, initial_features=2, elu=False):
        super().__init__()
        self.depth = depth

        n_features = [initial_features * 2 ** level
                      for level in range(self.depth)]
        # modules of the encoder path
        n_features_encode = [in_channels] + n_features
        # print("from encoder: n_features_encode", n_features_encode)
        if elu:
            self.encoder = nn.ModuleList(
                [self.conv_block_elu(n_features_encode[level],
                                     n_features_encode[
                                         level + 1])
                 for level in range(self.depth)])

            # the base convolution block
            n_features_base = n_features_encode[-1] * 2
            self.base = self.conv_block_elu(n_features_encode[-1],
                                            n_features_base)

        else:
            self.encoder = nn.ModuleList(
                [self.conv_block(n_features_encode[level],
                                 n_features_encode[
                                     level + 1])
                 for level in range(self.depth)])

            # the base convolution block
            n_features_base = n_features_encode[-1] * 2
            self.base = self.conv_block(n_features_encode[-1],
                                        n_features_base)

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList(
            [nn.MaxPool3d(2) for _ in range(self.depth)])

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
        return encoder_out, x


class UNetDecoder3D(nn.Module):
    def conv_block(self, in_channels, out_channels):
        # I have personally had better experience using relu instead of elu,
        # but this is worth confirming experimentally
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU())

    def conv_block_elu(self, in_channels, out_channels):
        # I have personally had better experience using relu instead of elu,
        # but this is worth confirming experimentally
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU())

    # upsampling via transposed 3d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels,
                                  kernel_size=2, stride=2)

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 depth: int = 4, initial_features: int = 16,
                 skip_connections: bool = True,
                 with_elu: bool = False,
                 final_activation: nn.Module or None = None) -> object:
        super().__init__()

        self.depth = depth
        self.skip_connections = skip_connections

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation,
                              nn.Module), "Activation must be torch module"

        n_features = [initial_features * 2 ** level for level in
                      range(self.depth)]
        n_features_encode = [in_channels] + n_features
        n_features_base = n_features_encode[-1] * 2
        n_features_decode = [n_features_base] + n_features[::-1]
        # print("from decoder: n_features_encode", n_features_encode)
        # print("from decoder: n_features_encode", n_features_decode)

        if self.skip_connections:
            # modules of the decoder path
            if with_elu:
                self.decoder = nn.ModuleList(
                    [self.conv_block_elu(n_features_decode[level],
                                         n_features_decode[level + 1]) for
                     level in range(self.depth)])
            else:
                self.decoder = nn.ModuleList(
                    [self.conv_block(n_features_decode[level],
                                     n_features_decode[level + 1]) for level in
                     range(self.depth)])
        else:
            if with_elu:
                self.decoder = nn.ModuleList(
                    [self.conv_block_elu(n_features_decode[level + 1],
                                         n_features_decode[level + 1]) for
                     level in
                     range(self.depth)])
            else:
                self.decoder = nn.ModuleList(
                    [self.conv_block(n_features_decode[level + 1],
                                     n_features_decode[level + 1]) for level in
                     range(self.depth)])

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

    def forward(self, input_tensor, encoder_outputs):

        decoder_outputs = []
        encoder_outputs = encoder_outputs[::-1]

        x = input_tensor
        for level in range(self.depth):
            if self.skip_connections:
                x = self.upsamplers[level](x)
                x = self.decoder[level](
                    self._crop_and_concat(x, encoder_outputs[level]))
            else:
                x = self.upsamplers[level](x)
                x = self.decoder[level](x)
            decoder_outputs.append(x)
            # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return decoder_outputs, x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, initial_features=64,
                 depth=4, final_activation=None, elu=False,
                 skip_connections=True):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_features = initial_features
        self.depth = depth
        self.final_activation = final_activation

        # contractive path
        self.encoder = UNetEncoder3D(in_channels=in_channels,
                                     depth=depth,
                                     initial_features=initial_features, elu=elu)
        # expansive path
        self.decoder = UNetDecoder3D(in_channels=in_channels,
                                     out_channels=out_channels,
                                     depth=depth,
                                     initial_features=initial_features,
                                     final_activation=final_activation,
                                     skip_connections=skip_connections,
                                     with_elu=elu)

    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = actions.crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward(self, inputs):
        # contractive path
        encoder_outputs, outputs = self.encoder(inputs)
        # expansive path
        decoder_outputs, final_outputs = self.decoder(outputs, encoder_outputs)
        return final_outputs

    def show(self, img):
        img = img.cpu()
        npimg = img.numpy()
        print("img.shape", img.shape)
        plt.imshow(npimg)
        plt.show()
        return

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, device,
                    print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.to(device)
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for batch_id, (x, y) in enumerate(loader):
            # move input and target to the active device (either cpu or gpu)
            x, y = x.to(device), y.to(device)

            # zero the gradients for this iteration
            optimizer.zero_grad()

            # apply model, calculate loss and run backwards pass
            prediction = self(x)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # To image:
            _, _, depth_dim, _, _ = prediction.shape
            sl = depth_dim // 2
            if batch_id % print_stats == 0:
                print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    (batch_id + 1) * len(x),
                    len(loader.dataset), 100. * (batch_id + 1) / len(loader),
                    loss.item()))
            if writer is not None:
                writer.add_scalar('train/loss-seg', loss.item(), epoch)

                if write_images:
                    # write images
                    x_im = vutils.make_grid(x[:, :, sl, :, :],
                                            normalize=True,
                                            scale_each=True)
                    y_im = vutils.make_grid(y[:, :, sl, :, :],
                                            normalize=True,
                                            scale_each=True)
                    y_pred_im = vutils.make_grid(
                        prediction[:, :, sl, :, :],
                        normalize=True,
                        scale_each=True)

                    writer.add_image('train/x', x_im, epoch)
                    writer.add_image('train/y', y_im, epoch)
                    writer.add_image('train/y_pred', y_pred_im, epoch)

        loss_avg = loss_cum / cnt
        print('Average Train Loss: {:.6f}'.format(loss_avg))
        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, device, writer=None,
                   write_images=False):
        # make sure network is on the device and in training mode
        self.to(device)
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        j_cum = 0.0
        a_cum = 0.0
        p_cum = 0.0
        r_cum = 0.0
        f_cum = 0.0
        cnt = 0

        # test loss
        for batch_id, (x, y) in enumerate(loader):
            # get the inputs
            x, y = x.to(device), y.to(device)

            # forward prop
            prediction = self(x)

            # compute loss
            loss = loss_fn(prediction, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # compute other interesting metrics
            # y_ = F.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0, ...]
            y_ = prediction.data.cpu().numpy()[:, 0, :, :, :]
            j_cum += jaccard(y_, y.cpu().numpy())
            a, p, r, f = accuracy_metrics(y_, y.cpu().numpy())
            a_cum += a
            p_cum += p
            r_cum += r
            f_cum += f
            print('[{}/{} ({:.0f}%)]\tValidation Loss: {:.6f}'.format(
                (batch_id + 1) * len(x),
                len(loader.dataset), 100. * (batch_id + 1) / len(loader), loss))
        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        j_avg = j_cum / cnt
        a_avg = a_cum / cnt
        p_avg = p_cum / cnt
        r_avg = r_cum / cnt
        f_avg = f_cum / cnt
        print('Average Validation Loss: {:.6f}'.format(loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss-seg', loss_avg, epoch)
            writer.add_scalar('test/jaccard', j_avg, epoch)
            writer.add_scalar('test/accuracy', a_avg, epoch)
            writer.add_scalar('test/precision', p_avg, epoch)
            writer.add_scalar('test/recall', r_avg, epoch)
            writer.add_scalar('test/f-score', f_avg, epoch)
            if write_images:
                # write images
                _, _, depth_dim, _, _ = prediction.shape
                sl = depth_dim // 2
                x_im = vutils.make_grid(x[:, :, sl, :, :],
                                        normalize=True,
                                        scale_each=True)
                y_im = vutils.make_grid(y[:, :, sl, :, :],
                                        normalize=True,
                                        scale_each=True)
                y_pred_im = vutils.make_grid(
                    prediction[:, :, sl, :, :],
                    normalize=True,
                    scale_each=True)
                writer.add_image('test/x', x_im.float(), epoch)
                writer.add_image('test/y', y_im.float(), epoch)
                writer.add_image('test/y_pred', y_pred_im.float(), epoch)

        return loss_avg

    def save_model(self, path_to_model: str, epoch: int, optimizer, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path_to_model)

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn, device, lr=1e-3,
                  step_size=1, gamma=1, epochs=100, test_freq=1, print_stats=1,
                  log_dir=None, write_images_freq=1, write_images=True):
        # log everything if necessary
        if log_dir is not None:
            writer = tb.SummaryWriter(log_dir)
            write_images = write_images
        else:
            writer = None
            write_images = False

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                              gamma=gamma)
        # scheduler = None
        test_loss_min = np.inf
        for epoch in range(epochs):

            print('Train Epoch %5d/%5d' % (epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_fn=loss_fn,
                             optimizer=optimizer, epoch=epoch, device=device,
                             print_stats=print_stats, writer=writer,
                             write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)
                if writer is not None:
                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate',
                                      float(scheduler.get_lr()[0]),
                                      epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn,
                                            device=device,
                                            epoch=epoch, writer=writer,
                                            write_images=write_images)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    if log_dir is not None:
                        best_checkpoint_path = join(log_dir,
                                                    'best_checkpoint.pytorch')
                        # torch.save(self, best_checkpoint_path)
                        self.save_model(path_to_model=best_checkpoint_path,
                                        epoch=epoch, optimizer=optimizer,
                                        loss=loss_fn)
            if log_dir is not None:
                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        if writer is not None:
            writer.close()


def unet_from_encoder_decoder(encoder, decoder):
    net = UNet3D(in_channels=encoder.in_channels,
                 out_channels=decoder.out_channels,
                 initial_features=encoder.initial_features, depth=encoder.depth)

    params = list(net.encoder.parameters())
    for i, param in enumerate(encoder.parameters()):
        params[i] = param

    params = list(net.decoder.parameters())
    for i, param in enumerate(decoder.parameters()):
        params[i] = param

    return net
