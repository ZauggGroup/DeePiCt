import torch

from tensors import actions
import numpy as np


def train_float(model, loader, optimizer, loss_function,
                epoch, device, log_interval=20, tb_logger=None):
    # set the model to train mode
    model.train()
    train_loss = 0
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.float(), y.float()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_function(prediction.float(), y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(x), len(loader.dataset),
                       100 * batch_id / len(loader), loss.item()))
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.log_scalar(tag='train_loss', value=loss.item(),
                                 step=step)

            log_image_interval = tb_logger.log_image_interval
            if step % log_image_interval == 0:
                # we always log the last validation images:
                img_indx, channel, size_z, size_y, size_z = x.shape
                single_tomo_shape = (1, 1, size_z, size_y, size_z)
                # we log four slices per cube:
                for slice_index in range(4):
                    slice_index *= size_z // 4
                    tb_logger.log_image(tag='val_input',
                                        image=actions.crop_tensor(
                                            x, single_tomo_shape)[
                                            0, 0, slice_index].to('cpu'),
                                        step=step)
                    tb_logger.log_image(tag='val_target',
                                        image=actions.crop_tensor(
                                            y, single_tomo_shape)[
                                            0, 0, slice_index].to('cpu'),
                                        step=step)
                    tb_logger.log_image(tag='val_prediction',
                                        image=
                                        actions.crop_tensor(
                                            prediction, single_tomo_shape)[
                                            0, 0, slice_index].to(
                                            'cpu'),
                                        step=step)

    train_loss /= len(loader)
    if tb_logger is not None:
        step = epoch * len(loader)
        tb_logger.log_scalar(tag='Average_train_loss', value=train_loss,
                             step=step)


def train(model, loader, optimizer, loss_function,
          epoch, device, log_interval=20, tb_logger=None, log_image=True,
          lr_scheduler=None):
    # set the model to train mode
    model.train()
    train_loss = 0
    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model, calculate loss and run backwards pass
        prediction = model(x)
        loss = loss_function(prediction, y.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # log to console
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(x),
                len(loader.dataset),
                       100 * batch_id / len(loader), loss.item()))

            # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.log_scalar(tag='train_loss', value=loss.item(),
                                 step=step)
            log_image_interval = tb_logger.log_image_interval
            if step % log_image_interval == 0:
                # we always log the last validation images:
                batch, channel, size_x, size_y, size_z = x.shape
                single_tomo_shape = (1, 1, size_x, size_y, size_z)
                for slice_index in range(1):
                    # log the image corresponding to:
                    #  batch, channel, x_slice =0, 0, slice_index
                    batch = 0
                    channel = 0  # logging the segmentation of background
                    slice_index *= size_z // 2
                    if log_image:
                        tb_logger.log_image(tag='val_input',
                                            image=actions.crop_tensor(
                                                x, single_tomo_shape)[
                                                batch, channel,
                                                slice_index].to(
                                                'cpu'),
                                            step=step)
                        if len(y.shape) == 5:
                            _, classes_to_log, _, _, _ = y.shape
                            for class_to_log in range(classes_to_log):
                                window_corner = (0, class_to_log, 0, 0, 0)
                                title_target = 'val_target class ' + str(
                                    class_to_log)
                                image = \
                                    actions.crop_window(y,
                                                        single_tomo_shape,
                                                        window_corner)
                                # print("image.shape = ", image.shape)
                                tb_logger.log_image(
                                    tag=title_target,
                                    image=image[0, 0, slice_index].to(
                                        'cpu'),
                                    step=step)
                                title_pred = 'val_pred class ' + str(
                                    class_to_log)
                                image_prediction = actions.crop_window(
                                    prediction, single_tomo_shape,
                                    window_corner)
                                tb_logger.log_image(
                                    tag=title_pred,
                                    image=image_prediction[
                                        0, 0, slice_index].to(
                                        'cpu'),
                                    step=step)
                        elif len(y.shape) == 4:
                            window_corner = (0, 0, 0, 0)
                            y_single_tomoshape = (1, size_x, size_y, size_z)
                            tb_logger.log_image(
                                tag='val_target class ' + str(channel),
                                image=
                                actions.crop_tensor(y, y_single_tomoshape)[
                                    batch, slice_index].to('cpu'),
                                step=step)
                            tb_logger.log_image(
                                tag='val_pred_class' + str(channel),
                                image=
                                actions.crop_window(
                                    prediction, single_tomo_shape,
                                    window_corner)[
                                    batch, channel, slice_index].to(
                                    'cpu'),
                                step=step)
                        else:
                            print("the size of the target tensor "
                                  "isnt loggable")
                    else:
                        print("Not logging images.")
    lr_scheduler.step(train_loss)

    train_loss /= len(loader)
    if tb_logger is not None:
        step = epoch * len(loader)
        tb_logger.log_scalar(tag='Average_train_loss', value=train_loss,
                             step=step)


def validate(model, loader, loss_function, metric, device, step=None,
             tb_logger=None, log_image_interval=None):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for index, data in enumerate(loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y.long()).item()
            val_metric += metric(prediction, y.long()).item()
            if tb_logger is not None:
                assert step is not None, \
                    "Need to know the current step to log validation results"
                if log_image_interval is not None:
                    if index % len(loader) == 0:
                        # we always log the last validation images
                        y = y.to('cpu').numpy()
                        nim, C, zdim, ydim, xdim = y.shape
                        max_sl = 0
                        max_level_sl = 0
                        for sl in range(zdim):
                            slide = y[0, 0, sl, ...]
                            label_sl = np.sum(slide > 0)
                            if max_level_sl < label_sl:
                                max_level_sl = label_sl
                                max_sl = sl
                        tb_logger.log_image(tag='val_input', image=x[0, :, max_sl, ...].to('cpu'), step=step)
                        tb_logger.log_image(tag='val_target', image=y[0, :, max_sl, ...], step=step)
                        tb_logger.log_image(tag='val_prediction', image=prediction[0, :, max_sl, ...].to('cpu'),
                                            step=step)

        # normalize loss and metric
        val_loss /= len(loader)
        val_metric /= len(loader)

        if tb_logger is not None:
            assert step is not None, \
                "Need to know the current step to log validation results"
            print('val_loss', "value=", val_loss, "step", step)
            tb_logger.log_scalar(tag='val_loss', value=val_loss, step=step)
            tb_logger.log_scalar(tag='val_metric', value=val_metric, step=step)
        print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(
            val_loss, val_metric))
    return val_loss


def compute_global_dice(model, loader, device, overlap):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    numerator = 0
    denominator = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for index, data in enumerate(loader):
            x, y = data
            mask = np.zeros(x.shape)
            ones = np.ones(x.shape)
            mask[:, :, overlap:-overlap, overlap:-overlap, overlap:-overlap] = ones[:, :, overlap:-overlap,
                                                                               overlap:-overlap,
                                                                               overlap:-overlap]
            mask = torch.from_numpy(mask).to(device)
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            numerator += (prediction * y.long() * mask).sum()
            denominator += (prediction * prediction * mask).sum() + (y.long() * y.long() * mask).sum()

        # normalize loss and metric
        print(numerator, denominator)
        global_dice = 2 * numerator.item() / denominator.item()

        print('\nValidate: Global_dice: {:.4f}'.format(global_dice))
    return global_dice


def validate_float(model, loader, loss_function, metric, device, step=None,
                   tb_logger=None):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():

        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.float(), y.float()
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction.float(), y.float()).item()
            val_metric += metric(prediction.float(),
                                 actions.crop_tensor(y.float(),
                                                     prediction.shape)).item()

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    if tb_logger is not None:
        assert step is not None, \
            "Need to know the current step to log validation results"
        tb_logger.log_scalar(tag='val_loss', value=val_loss, step=step)
        tb_logger.log_scalar(tag='val_metric', value=val_metric, step=step)

    print(
        '\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(
            val_loss, val_metric))

    return val_loss
