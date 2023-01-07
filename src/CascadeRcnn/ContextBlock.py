import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeUniform, HeNormal, Constant

def last_zero_init(m):
    if isinstance(m, nn.SequentialCell):
        m[-1].weight = initializer(Constant(0), m[-1].weight.shape, mindspore.float32)
    else:
        m.weight = initializer(Constant(0), m.weight.shape, mindspore.float32)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            module.weight = initializer(HeUniform(a, mode, nonlinearity), module.weight.shape, mindspore.float32)
        else:
            module.weight = initializer(HeNormal(a, mode, nonlinearity), module.weight.shape, mindspore.float32)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias = initializer(Constant(bias), module.bias.shape, mindspore.float32)

class ContextBlock(nn.Cell):
    """ContextBlock module in GCNet.

    See 'GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond'
    (https://arxiv.org/abs/1904.11492) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        ratio (float): Ratio of channels of transform bottleneck
        pooling_type (str): Pooling method for context modeling.
            Options are 'att' and 'avg', stand for attention pooling and
            average pooling respectively. Default: 'att'.
        fusion_types (Sequence[str]): Fusion method for feature fusion,
            Options are 'channels_add', 'channel_mul', stand for channelwise
            addition and multiplication respectively. Default: ('channel_add',)
    """

    _abbr_ = 'context_block'

    def __init__(self,
                 in_channels,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super().__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1, has_bias=True)
            self.softmax = nn.Softmax(axis=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.SequentialCell(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1, has_bias=True),
                nn.LayerNorm([self.planes, 1, 1], begin_norm_axis=1, begin_params_axis=1),
                nn.ReLU(),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1, has_bias=True))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.SequentialCell(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1, has_bias=True),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1, has_bias=True))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.shape
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            expand_op = P.ExpandDims()
            input_x = expand_op(input_x, 1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = expand_op(context_mask, -1)
            # [N, 1, C, 1]
            context = ops.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def construct(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            sigmoid_op = P.Sigmoid()
            channel_mul_term = sigmoid_op(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out