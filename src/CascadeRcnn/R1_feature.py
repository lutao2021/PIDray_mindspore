# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CascadeRcnn feature pyramid network."""

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from .ContextBlock import ContextBlock


def bias_init_zeros(shape):
    """Bias init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)))


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer("XavierUniform", shape=shape, dtype=mstype.float32).init_data()
    shape_bias = (out_channels,)
    biass = bias_init_zeros(shape_bias)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)


class channel_attention(nn.Cell):
    def __init__(self):
        super(channel_attention,self).__init__()
        M = 5
        d = 128
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.SequentialCell(nn.Conv2d(256, d, kernel_size=1, stride=1, has_bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU())
        self.fcs = nn.CellList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, 256, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(axis=1)

    def construct(self, inputs, num_levels):
        sum_op = P.ReduceSum()
        feats_U = sum_op(inputs, 1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        concat_op = P.Concat(1)
        attention_vectors = concat_op(attention_vectors)
        attention_vectors = attention_vectors.view(-1, num_levels, 256, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        return attention_vectors


class spatial_attention(nn.Cell):
    def __init__(self, kernel_size=7):
        super(spatial_attention,self).__init__()
        M = 5
        d = 128
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.CellList([])
        for i in range(M):
            self.conv.append(
                 nn.Conv2d(2, 1, kernel_size, padding=padding, has_bias=False, pad_mode='pad')
            )
        self.softmax = nn.Softmax(axis=1)

    def construct(self, inputs, num_levels):
        sum_op = P.ReduceSum()
        inputs = sum_op(inputs, 1)
        mean_op = P.ReduceMean(keep_dims=True)
        avg_out = mean_op(inputs, 1)
        max_op = P.ArgMaxWithValue(keep_dims=True, axis=1)
        _, max_out = max_op(inputs)
        concat_op = P.Concat(1)
        x = concat_op([avg_out, max_out])

        attention_vectors = [conv(x) for conv in self.conv]
        attention_vectors = concat_op(attention_vectors)
        attention_vectors = self.softmax(attention_vectors)
        expand_op = P.ExpandDims()
        attention_vectors = expand_op(attention_vectors, 2)
        return attention_vectors


class selective_attention(nn.Cell):

    def __init__(self, refine_level):
        super(selective_attention, self).__init__()
        self.refine_level = refine_level
        self.channel_att = channel_attention()
        self.spatial_att = spatial_attention()
        self.refine = ContextBlock(256, 1. / 16)
        # self.refine = ContextBlock(256, 1. / 2)

    def construct(self, inputs):
        feats = []
        gather_size = inputs[self.refine_level].shape[2:]
        num_levels = len(inputs)
        for i in range(num_levels):
            if i < self.refine_level:
                # adaptive_maxpool2d = nn.AdaptiveMaxPool2d(gather_size)
                adaptive_maxpool2d = P.MaxPool(2**(self.refine_level - i), 2**(self.refine_level - i))
                gathered = adaptive_maxpool2d(inputs[i])
            else:
                interpolate_nearest_op = ops.ResizeNearestNeighbor(gather_size)
                gathered = interpolate_nearest_op(inputs[i])
            feats.append(gathered)
        concat_op = P.Concat(1)
        feats = concat_op(feats)
        feats = feats.view(feats.shape[0], num_levels, 256, feats.shape[2], feats.shape[3])

        channel_attention_vectors = self.channel_att(feats, num_levels)
        sum_op = P.ReduceSum(keep_dims=True)
        feats_C = sum_op(feats * channel_attention_vectors, 1)

        spatial_attention_vectors = self.spatial_att(feats, num_levels)
        # return [spatial_attention_vectors[:,0,:,:,:],spatial_attention_vectors[:,1,:,:,:],spatial_attention_vectors[:,2,:,:,:],
        #         spatial_attention_vectors[:,3,:,:,:],spatial_attention_vectors[:,4,:,:,:]]
        feats_S = sum_op(feats * spatial_attention_vectors, 1)

        feats_sum = feats_C + feats_S
        # feats_sum = feats_S
        # bsf = feats_sum
        feats_sum = ops.squeeze(feats_sum)
        bsf = self.refine(feats_sum)

        # residual = nn.AdaptiveMaxPool2d(gather_size)(bsf)
        residual = bsf
        return residual + inputs[self.refine_level]



class FeatPyramidNeck(nn.Cell):
    """
    Feature pyramid network cell, usually uses as network neck.

    Applies the convolution on multiple, input feature maps
    and output feature map with same channel size. if required num of
    output larger then num of inputs, add extra maxpooling for further
    downsampling;

    Args:
        in_channels (tuple) - Channel size of input feature maps.
        out_channels (int) - Channel size output.
        num_outs (int) - Num of output features.

    Returns:
        Tuple, with tensors of same channel size.

    Examples:
        neck = FeatPyramidNeck([100,200,300], 50, 4)
        input_data = (normal(0,0.1,(1,c,1280//(4*2**i), 768//(4*2**i)),
                      dtype=np.float32) \
                      for i, c in enumerate(config.fpn_in_channels))
        x = neck(input_data)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 feature_shape):
        super(FeatPyramidNeck, self).__init__()
        self.num_outs = num_outs
        self.in_channels = in_channels
        self.fpn_layer = len(self.in_channels)

        assert not self.num_outs < len(in_channels)

        self.lateral_convs_list_ = []
        self.fpn_convs_ = []

        for _, channel in enumerate(in_channels):
            l_conv = _conv(channel, out_channels, kernel_size=1, stride=1, padding=0, pad_mode='valid')
            fpn_conv = _conv(out_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same')
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        # self.interpolate1 = P.ResizeNearestNeighbor((48, 80))
        # self.interpolate2 = P.ResizeNearestNeighbor((96, 160))
        # self.interpolate3 = P.ResizeNearestNeighbor((192, 320))
        self.interpolate1 = P.ResizeNearestNeighbor(feature_shape[2])
        self.interpolate2 = P.ResizeNearestNeighbor(feature_shape[1])
        self.interpolate3 = P.ResizeNearestNeighbor(feature_shape[0])
        self.maxpool = P.MaxPool(kernel_size=1, strides=2, pad_mode="same")

        self.selective_attention_0 = selective_attention(0)
        self.selective_attention_1 = selective_attention(1)
        self.selective_attention_2 = selective_attention(2)
        self.selective_attention_3 = selective_attention(3)
        self.selective_attention_4 = selective_attention(4)

    def enhance_feature(self, inputs):
        output = []
        output_0 = self.selective_attention_0(inputs)
        output_1 = self.selective_attention_1(inputs)
        output_2 = self.selective_attention_2(inputs)
        output_3 = self.selective_attention_3(inputs)
        output_4 = self.selective_attention_4(inputs)
        output.append(output_0)
        output.append(output_1)
        output.append(output_2)
        output.append(output_3)
        output.append(output_4)
        return tuple(output)

    def construct(self, inputs):
        "FPN neck"
        x = ()
        for i in range(self.fpn_layer):
            x += (self.lateral_convs_list[i](inputs[i]),)

        y = (x[3],)
        y = y + (x[2] + self.interpolate1(y[self.fpn_layer - 4]),)
        y = y + (x[1] + self.interpolate2(y[self.fpn_layer - 3]),)
        y = y + (x[0] + self.interpolate3(y[self.fpn_layer - 2]),)

        z = ()
        for i in range(self.fpn_layer - 1, -1, -1):
            z = z + (y[i],)

        outs = ()
        for i in range(self.fpn_layer):
            outs = outs + (self.fpn_convs_list[i](z[i]),)

        for i in range(self.num_outs - self.fpn_layer):
            outs = outs + (self.maxpool(outs[3]),)

        outs = self.enhance_feature(outs)
        return outs
