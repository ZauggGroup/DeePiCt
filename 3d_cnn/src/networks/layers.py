import torch
import torch.nn as nn


# implementation of the 3d group normalization layer
# (https://arxiv.org/abs/1803.08494)
class GroupNorm3d(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm3d, self).__init__()

        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        assert channels % self.num_groups == 0

        x = x.view(batch_size, self.num_groups, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(batch_size, channels, depth, height, width)

        return x * self.weight + self.bias
