import tensorboardX as tb
import torch


# we adapt the tensorboard logger s.t. it can also log images
class TensorBoard(object):
    """
    Adapted version from A. Kreshuk and C. Pape
    """

    def __init__(self, log_dir: str, log_image_interval: int):
        self.log_dir = log_dir
        self.log_image_interval = log_image_interval
        self.writer = tb.SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def log_image(self, tag, image, step):

        # convert to numpy array
        if torch.is_tensor(image):
            image = image.detach().numpy()

        # change the image normalization for the tensorboard logger
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image /= max_val

        self.writer.add_image(tag, img_tensor=image, global_step=step)


class TensorBoard_multiclass(object):
    """
    Adapted version from A. Kreshuk and C. Pape
    """

    def __init__(self, log_dir: str, log_image_interval: int):
        self.log_dir = log_dir
        self.log_image_interval = log_image_interval
        self.writer = tb.SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def log_image(self, tag, image, step):
        # print("the method is not yet working")
        # convert to numpy array
        if torch.is_tensor(image):
            image = image.detach().numpy()

        # change the image normalization for the tensorboard logger
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image = image / max_val

        self.writer.add_image(tag, img_tensor=image, global_step=step)
