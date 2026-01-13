import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, constant_


zeros_initializer = partial(constant_, val=0.0)  # 神经网络开始学习前，默认将某些参数（如偏置项 $b$）设为 0

# 高斯模糊（处理原子距离）--将距离变为信号
def gaussian_smearing(distances, offset, widths, centered=False):

    if not centered:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)  # 计算高斯函数的衰减系数
        diff = distances - offset  # 计算原子间实际距离与预设“刻度”的偏差

    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances

    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))  # 核心公式，它把一个距离数值，映射成一组 0 到 1 之间的概率分布

    return gauss

# 对上述函数的封装，使其成为神经网络的一个标准层
class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    sample struct dictionary:

        struct = {'start': 0.0, 'stop':5.0, 'n_gaussians': 32, 'centered': False, 'trainable': False}

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(
        self, start, stop, n_gaussians, width=None, centered=False, trainable=False
    ):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor(
                (offset[1] - offset[0]) * torch.ones_like(offset)
            )
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        result = gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )

        return result

# 神经网络的“神经元”
class Dense(nn.Linear):
    """Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init  # 初始化。在训练开始前，随机给神经网络里的成千上万个参数分发初始数字（比如使用 Xavier 分布）
        self.bias_init = bias_init  # 从 PyTorch 框架中继承基础的线性层（即矩阵乘法 y = Wx + b）
        self.activation = activation  # 定义“激活函数”。它负责给数学函数加入非线性特征

        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        y = super().forward(inputs)
        if self.activation:
            y = self.activation(y)

        return y
