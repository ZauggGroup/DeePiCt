# Implementation inspired from:
# https://github.com/JorisRoels/domain-adaptive-segmentation
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from src.python.networks.unet_new import UNetEncoder3D, UNetDecoder3D
from src.python.networks.unet_new import unet_from_encoder_decoder


class YNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, initial_features=64,
                 depth=4, segm_final_activation=nn.Sigmoid, lambda_rec=1e-3):
        super(YNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_features = initial_features
        self.depth = depth
        self.lambda_rec = lambda_rec

        # encoder
        self.encoder = UNetEncoder3D(in_channels=in_channels,
                                     initial_features=initial_features,
                                     depth=depth)

        # segmentation decoder
        self.segmentation_decoder = \
            UNetDecoder3D(out_channels=in_channels,
                          initial_features=initial_features,
                          depth=depth, final_activation=segm_final_activation,
                          skip_connections=True)

        # reconstruction decoder
        self.reconstruction_decoder = \
            UNetDecoder3D(out_channels=in_channels,
                          initial_features=initial_features,
                          depth=depth, final_activation=None,
                          skip_connections=False)

    def forward(self, inputs):
        # encoder
        inputs = inputs.float()
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        _, segmentation_outputs = \
            self.segmentation_decoder(encoded, encoder_outputs)

        # reconstruction decoder

        _, reconstruction_outputs = \
            self.reconstruction_decoder(encoded, encoder_outputs)
        return reconstruction_outputs, segmentation_outputs

    def get_segmentation_net(self):
        return unet_from_encoder_decoder(self.encoder,
                                         self.segmentation_decoder)

    def train_epoch(self, loader_src, loader_tar,
                    optimizer, loss_seg_fn, loss_rec_fn, epoch, device,
                    print_stats=1, writer=None, write_images=False):
        self.to(device)
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for batch_id, (x_src, y_src) in enumerate(loader_src):
            # move input and target to the active device (either cpu or gpu)
            _, (x_tar, _) = list_tar[batch_id]

            x_src, y_src, x_tar = \
                x_src.to(device), y_src.to(device), x_tar.to(device)
            x_src, y_src, x_tar = x_src.float(), y_src.float(), x_tar.float()
            # zero the gradients for this iteration
            optimizer.zero_grad()
            x_src_pred, y_src_pred = self(x_src)
            x_tar_pred, y_tar_pred = self(x_tar)

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_rec = 0.5 * (
                loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            total_loss = loss_seg + self.lambda_rec * loss_rec
            loss_seg_cum += loss_seg.detach().cpu().numpy()
            loss_rec_cum += loss_rec.detach().cpu().numpy()
            total_loss_cum += total_loss.detach().cpu().numpy()

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if batch_id % print_stats == 0:
                print(
                    'Epoch %5d - Iteration %5d/%5d - Train Loss seg: %.6f - Train Loss rec: %.6f - Train Loss: %.6f'
                    % (epoch + 1, batch_id + 1,
                       len(loader_src.dataset) / loader_src.batch_size,
                       loss_seg, loss_rec, total_loss))

            # don't forget to compute the average and print it
            loss_seg_avg = loss_seg_cum / len(loader_src.dataset)
            loss_rec_avg = loss_rec_cum / len(loader_src.dataset)
            total_loss_avg = total_loss_cum / len(loader_src.dataset)
            print(
                'Epoch %5d - Train Loss seg: %.6f - Train Loss rec: %.6f - Train Loss: %.6f'
                % (epoch + 1, loss_seg_avg, loss_rec_avg, total_loss_avg))

            # log everything
            if writer is not None and write_images:
                # scalars
                writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
                writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
                writer.add_scalar('train/loss', total_loss_avg, epoch)

                # images
                _, _, depth_dim, _, _ = y_src_pred.shape
                sl = depth_dim // 2
                x = torch.cat((x_src, x_tar), dim=0)
                x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
                y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
                x = vutils.make_grid(x[:, :, sl, :, :],
                                     normalize=True, scale_each=True)
                ys = vutils.make_grid(y_src[:, :, sl, :, :],
                                      normalize=True, scale_each=True)
                x_pred = vutils.make_grid(x_pred[:, :, sl, :, :],
                                          normalize=True, scale_each=True)
                y_pred = vutils.make_grid(
                    # F.softmax(y_pred, dim=1)[0:1, :, sl, :, :],
                    y_pred[:, :, sl, :, :], normalize=True, scale_each=True)
                writer.add_image('train/x', x.float(), epoch)
                writer.add_image('train/y', ys.float(), epoch)
                writer.add_image('train/x-pred', x_pred.float(), epoch)
                writer.add_image('train/y-pred', y_pred.float(), epoch)

            return total_loss_avg

    def test_epoch(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn,
                   epoch, device, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.to(device)
        self.eval()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        # loss_seg_tar_cum = 0.0  # todo... use this?
        loss_rec_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for batch_id, (x_src, y_src) in enumerate(loader_src):
            # move input and target to the active device (either cpu or gpu)
            _, (x_tar, _) = list_tar[batch_id]

            x_src, y_src, x_tar = \
                x_src.to(device), y_src.to(device), x_tar.to(device)
            x_src, y_src, x_tar = x_src.float(), y_src.float(), x_tar.float()
            x_src, y_src, x_tar = x_src.float(), y_src.float(), x_tar.float()

            x_src_pred, y_src_pred = self.forward(x_src)
            x_tar_pred, y_tar_pred = self.forward(x_tar)

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_rec = 0.5 * (
                loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            total_loss = loss_seg + self.lambda_rec * loss_rec
            loss_seg_cum += loss_seg.detach().cpu().numpy()
            loss_rec_cum += loss_rec.detach().cpu().numpy()
            total_loss_cum += total_loss.detach().cpu().numpy()

            # backward prop
            total_loss.backward()

            # don't forget to compute the average and print it
            loss_seg_avg = loss_seg_cum / len(loader_src.dataset)
            loss_rec_avg = loss_rec_cum / len(loader_src.dataset)
            total_loss_avg = total_loss_cum / len(loader_src.dataset)
            print(
                'Epoch %5d - Test Loss seg: %.6f - Test Loss rec: %.6f - Test Loss: %.6f'
                % (epoch + 1, loss_seg_avg, loss_rec_avg, total_loss_avg))

            # log everything
            if writer is not None and write_images:
                # scalars
                writer.add_scalar('test/loss-seg', loss_seg_avg, epoch)
                writer.add_scalar('test/loss-rec', loss_rec_avg, epoch)
                writer.add_scalar('test/loss', total_loss_avg, epoch)
                # images
                _, _, depth_dim, _, _ = y_src_pred.shape
                sl = depth_dim // 2
                x = torch.cat((x_src, x_tar), dim=0)
                x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
                y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
                print("x.shape = ", x.shape)
                print("x_pred.shape = ", x_pred.shape)
                print("y_pred.shape = ", y_pred.shape)
                x = vutils.make_grid(x[:, :, sl, :, :],
                                     normalize=True, scale_each=True)
                y = vutils.make_grid(y_src[:, :, sl, :, :],
                                      normalize=True, scale_each=True)
                x_pred = vutils.make_grid(x_pred[:, :, sl, :, :],
                                          normalize=True, scale_each=True)
                y_pred = vutils.make_grid(y_pred[:, :, sl, :, :],
                                          normalize=True, scale_each=True)
                writer.add_image('test/x', x.float(), epoch)
                writer.add_image('test/y', y.float(), epoch)
                writer.add_image('test/x-pred', x_pred.float(), epoch)
                writer.add_image('test/y-pred', y_pred.float(), epoch)

            return total_loss_avg

    def train_net(self, train_loader_source, train_loader_target,
                  test_loader_source, test_loader_target,
                  optimizer, loss_seg_fn, loss_rec_fn, device, scheduler=None,
                  epochs=100, test_freq=1, print_stats=1,
                  log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('Epoch %5d/%5d' % (epoch + 1, epochs))

            # train the model for one epoch
            self.train_epoch(loader_src=train_loader_source,
                             loader_tar=train_loader_target,
                             optimizer=optimizer, loss_seg_fn=loss_seg_fn,
                             loss_rec_fn=loss_rec_fn, epoch=epoch,
                             device=device, print_stats=print_stats,
                             writer=writer,
                             write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]),
                                  epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader_src=test_loader_source,
                                            loader_tar=test_loader_target,
                                            loss_seg_fn=loss_seg_fn,
                                            loss_rec_fn=loss_rec_fn,
                                            epoch=epoch, device=device,
                                            writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self,
                               os.path.join(log_dir, 'best_checkpoint.pytorch'))
                    # ToDo add save_model method
            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()
