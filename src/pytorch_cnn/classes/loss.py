import torch
import torch.nn as nn
import torch.nn.functional as F

from tensors.actions import crop_tensor


class BCELoss(nn.Module):
    def __init__(self, weight_function=None):
        # in torch, loss functions must inherit from `torch.nn.Module`
        # and call the super constructor to be compatible with the
        # automatic differentiation capabilities
        super().__init__()

        # the weighting function is optional and will not
        # be used if it is `None`
        # if a weighting function is given, it must
        # take the target tensor as input and return
        # a weight tensor with the same shape
        self.weight_function = weight_function

    # to implement a loss function, we only need to
    # overload the forward pass.
    # the backward pass will be performed by torch automatically
    def forward(self, input, target):
        ishape = input.shape
        tshape = target.shape

        # make sure that the batches and channels target and input agree
        assert ishape[:2] == tshape[:2]
        assert ishape[1] == 1, "Only supports a single channel for now"

        # crop the target to fit the input
        target = crop_tensor(target, ishape)

        # check if we have a weighting function and if so apply it
        if self.weight_function is not None:
            # apply the weight function
            weight = self.weight_function(target)
            # compute the loss WITHOUT reduction, which means that
            # the los will have the same shape as input and target
            loss = F.binary_cross_entropy(input, target, reduction='none')

            # multiply the loss by the weight and
            # reduce it via element-wise mean
            assert weight.shape == loss.shape, "Loss and weight must have the same shape"
            loss = torch.mean(loss * weight)

        # if we don't have a weighting function, just apply the loss
        else:
            loss = F.binary_cross_entropy(input, target)
        return loss


class Multi_class_CELoss(nn.Module):
    def __init__(self, in_channels, weight_function=None):
        # in torch, loss functions must inherit from `torch.nn.Module`
        # and call the super constructor to be compatible with the
        # automatic differentiation capabilities
        super().__init__()
        self.in_channels = in_channels
        # the weighting function is optional and will not
        # be used if it is `None`
        # if a weighting function is given, it must
        # take the target tensor as input and return
        # a weight tensor with the same shape
        self.weight_function = weight_function

    # to implement a loss function, we only need to
    # overload the forward pass.
    # the backward pass will be performed by torch automatically
    def forward(self, input, target):
        ishape = input.shape
        tshape = target.shape

        # make sure that the batches and channels target and input agree
        assert ishape[:2] == tshape[:2]
        assert ishape[1] == self.in_channels, "check number of input channels"

        # crop the target to fit the input
        target = crop_tensor(target, ishape)

        # check if we have a weighting function and if so apply it
        if self.weight_function is not None:
            # apply the weight function
            weight = self.weight_function(target)
            # compute the loss WITHOUT reduction, which means that
            # the loss will have the same shape as input and target
            loss = F.binary_cross_entropy(input, target, reduction='none')

            # multiply the loss by the weight and
            # reduce it via element-wise mean
            assert weight.shape == loss.shape, "Loss and weight must have the same shape"
            loss = torch.mean(loss * weight)

        # if we don't have a weighting function, just apply the loss
        else:
            loss = F.binary_cross_entropy(input, target)
        return loss


# sorensen dice coefficient implemented in torch
# the coefficient takes values in [0, 1], where 0 is
# the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        denominator = prediction.sum() + target.sum()
        return 2 * intersection / denominator.clamp(min=self.eps)


# class DiceCoefficientLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#
#     # the dice coefficient of two sets represented as vectors a, b ca be
#     # computed as (2 *|a b| / (a^2 + b^2))
#     def forward(self, prediction, target):
#         prediction.float()
#         target.float()
#         intersection = (prediction * target).sum()
#         denominator = (prediction * prediction).sum() + (target * target).sum()
#         return 1 - (2 * intersection / denominator.clamp(min=self.eps))


class DiceCoefficientLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # the dice coefficient of two sets represented as vectors a, b can be
        # computed as (2 *|a b| / (a^2 + b^2))

    def forward(self, input, target):
        input.float()
        target.float()
        dice_loss = 0
        channels = input.shape[1]
        # print("channels = ", channels)
        # print("input.shape", input.shape)
        for channel in range(channels):
            channel_prediction = input[:, channel, ...].float()
            channel_target = target[:, channel, ...].float()
            intersection = (channel_prediction * channel_target).sum()
            denominator = (channel_prediction * channel_prediction).sum() + (
                channel_target * channel_target).sum()
            dice_loss += (1 - 2 * intersection / denominator.clamp(
                min=self.eps))
        dice_loss /= channels  # normalize loss
        return dice_loss


class DiceCoefficientLoss_multilabel(nn.Module):
    def __init__(self, eps=1e-6, weight=None):
        super().__init__()
        self.eps = eps
        # the dice coefficient of two sets represented as vectors a, b can be
        # computed as (2 *|a b| / (a^2 + b^2))

    def forward(self, input, target):
        input.float()
        target.float()
        dice_loss = 0
        channels = input.shape[1]
        # print("channels = ", channels)
        # print("input.shape", input.shape)
        for channel in range(channels):
            channel_prediction = input[:, channel, ...].float()
            channel_target = target[:, channel, ...].float()
            intersection = (channel_prediction * channel_target).sum()
            denominator = (channel_prediction * channel_prediction).sum() + (
                channel_target * channel_target).sum()
            dice_loss += (1 - 2 * intersection / denominator.clamp(
                min=self.eps))
        dice_loss /= channels  # normalize loss
        return dice_loss


class TanhDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # the dice coefficient of two sets represented as vectors a, b can be
        # computed as (2 *|a b| / (a^2 + b^2))

    def forward(self, input, target):
        input.float()
        target.float()
        dice_loss = 0
        channels = input.shape[1]
        print(input.shape)
        tanh = torch.nn.Hardtanh(min_val=0, max_val=0.5, inplace=False)
        for channel in range(channels):
            channel_prediction = tanh(input[:, channel, ...].float()).float()
            channel_target = target[:, channel, ...].float()
            intersection = (channel_prediction * channel_target).sum()
            denominator = (channel_prediction * channel_prediction).sum() + (
                channel_target * channel_target).sum()
            dice_loss += (1 - 2 * intersection / denominator.clamp(
                min=self.eps))
        dice_loss /= channels  # normalize loss
        return dice_loss


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.mean((input.float() - target.float()) ** 2)


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.sum((input - target) ** 2) / torch.sum(target ** 2)
