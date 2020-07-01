import torch

from src.python.tensors import actions


def train_float(model, loader, optimizer, loss_function,
                epoch, device, log_interval=20, tb_logger=None):
    # set the model to train mode
    model.train()
    train_loss = 0
    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.float(), y.float()
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model, calculate loss and run backwards pass
        prediction = model(x)
        loss = loss_function(prediction.float(), y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # log to console
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(x),
                len(loader.dataset),
                       100. * batch_id / len(loader), loss.item()))

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.log_scalar(tag='train_loss', value=loss.item(), step=step)

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
          epoch, device, log_interval=20, tb_logger=None, log_image=True):
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
                       100. * batch_id / len(loader), loss.item()))

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.log_scalar(tag='train_loss', value=loss.item(), step=step)

            log_image_interval = tb_logger.log_image_interval
            if step % log_image_interval == 0:
                # we always log the last validation images:
                batch, channel, size_x, size_y, size_z = x.shape
                # print("y.shape = ", y.shape)
                # print("x.shape = ", x.shape)
                single_tomo_shape = (1, 1, size_x, size_y, size_z)
                # we log four slices per cube:
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
                                                batch, channel, slice_index].to(
                                                'cpu'),
                                            step=step)

                        if len(y.shape) == 5:
                            _, classes_to_log, _, _, _ = y.shape
                            for class_to_log in range(classes_to_log):
                                window_corner = (0, class_to_log, 0, 0, 0)
                                title_target = 'val_target class ' + str(
                                    class_to_log)
                                image = actions.crop_window(y,
                                                            single_tomo_shape,
                                                            window_corner)
                                # print("image.shape = ", image.shape)

                                tb_logger.log_image(
                                    tag=title_target,
                                    image=image[0, 0, slice_index].to('cpu'),
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
                            print("the size of the target tensor isnt loggable")
                    else:
                        print("Not logging images.")


    train_loss /= len(loader)
    if tb_logger is not None:
        step = epoch * len(loader)
        tb_logger.log_scalar(tag='Average_train_loss', value=train_loss,
                             step=step)


def validate(model, loader, loss_function, metric, device, step=None,
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
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y.long()).item()
            val_metric += metric(prediction, y.long()).item()

    # normalize loss and metric
    val_loss /= len(loader.dataset)
    val_metric /= len(loader.dataset)

    if tb_logger is not None:
        assert step is not None, \
            "Need to know the current step to log validation results"
        tb_logger.log_scalar(tag='val_loss', value=val_loss, step=step)
        tb_logger.log_scalar(tag='val_metric', value=val_metric, step=step)
        # we always log the last validation images
        # pshape = prediction.shape
        # tb_logger.log_image(tag='val_input', image=crop_tensor(x, pshape)[0, 0].to('cpu'), step=step)
        # tb_logger.log_image(tag='val_target', image=crop_tensor(y, pshape)[0, 0].to('cpu'), step=step)
        # tb_logger.log_image(tag='val_prediction', image=prediction[0, 0].to('cpu'), step=step)

    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(
        val_loss, val_metric))
    return val_loss


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
    val_loss /= len(loader.dataset)
    val_metric /= len(loader.dataset)

    if tb_logger is not None:
        assert step is not None, \
            "Need to know the current step to log validation results"
        tb_logger.log_scalar(tag='val_loss', value=val_loss, step=step)
        tb_logger.log_scalar(tag='val_metric', value=val_metric, step=step)

    print(
        '\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(
            val_loss, val_metric))

    return val_loss
    # build default-unet with sigmoid activation
    # to normalize prediction to [0, 1]
