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
"""CascadeRcnn based on ResNet101."""
import mindspore
import mindspore.ops as ops
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from .resnet101 import ResNetFea, ResidualBlockUsing
from .bbox_assign_sample_stage2 import BboxAssignSampleForRcnn
from .bbox_assign_sample_stage2_1_1 import BboxAssignSampleForRcnn_1_1
from .bbox_assign_sample_stage2_2_1 import BboxAssignSampleForRcnn_2_1
# from .fpn_neck import FeatPyramidNeck
from .R1_feature import FeatPyramidNeck
from .proposal_generator import Proposal
from .rcnn import Rcnn
from .rcnn1 import Rcnn_1
from .rcnn2 import Rcnn_2
from .rpn import RPN
from .roi_align import SingleRoIExtractor
from .roi_align_1 import SingleRoIExtractor_1
from .roi_align_2 import SingleRoIExtractor_2
from .anchor_generator import AnchorGenerator
from .transformer import Transformer
from .SinePositinoalEncoding import SinePositionalEncoding


class Cascade_Rcnn_Resnet101(nn.Cell):
    """
    CascadeRcnn Network.

    Note:
        backbone =

    Returns:
        Tuple, tuple of output tensor.
        rpn_loss: Scalar, Total loss of RPN subnet.
        rcnn_loss: Scalar, Total loss of RCNN subnet.
        rpn_cls_loss: Scalar, Classification loss of RPN subnet.
        rpn_reg_loss: Scalar, Regression loss of RPN subnet.
        rcnn_cls_loss: Scalar, Classification loss of RCNN subnet.
        rcnn_reg_loss: Scalar, Regression loss of RCNN subnet.

    Examples:
        net = Cascade_Rcnn_()
    """
    def __init__(self, config):
        super(Cascade_Rcnn_Resnet101, self).__init__()
        self.dtype = np.float32
        self.ms_type = mstype.float32
        self.train_batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_strides = config.anchor_strides
        self.target_means = tuple(config.rcnn_target_means)
        self.target_stds = tuple(config.rcnn_target_stds)
        self.img_h = config.img_height
        self.img_w = config.img_width

        # Anchor generator
        anchor_base_sizes = None
        self.anchor_base_sizes = list(
            self.anchor_strides) if anchor_base_sizes is None else anchor_base_sizes

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        featmap_sizes = config.feature_shapes
        assert len(featmap_sizes) == len(self.anchor_generators)

        self.anchor_list = self.get_anchors(featmap_sizes)

        # Backbone resnet101
        self.backbone = ResNetFea(ResidualBlockUsing,
                                  config.resnet_block,
                                  config.resnet_in_channels,
                                  config.resnet_out_channels,
                                  False)

        # Fpn
        self.R1_feature = FeatPyramidNeck(config.fpn_in_channels,
                                        config.fpn_out_channels,
                                        config.fpn_num_outs,
                                        config.feature_shapes)

        # Rpn and rpn loss
        self.gt_labels_stage1 = Tensor(np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8))
        self.rpn_with_loss = RPN(config,
                                 self.train_batch_size,
                                 config.rpn_in_channels,
                                 config.rpn_feat_channels,
                                 config.num_anchors,
                                 config.rpn_cls_out_channels)

        # Proposal
        self.proposal_generator = Proposal(config,
                                           self.train_batch_size,
                                           config.activate_num_classes,
                                           config.use_sigmoid_cls)
        self.proposal_generator.set_train_local(config, True)
        self.proposal_generator_test = Proposal(config,
                                                config.test_batch_size,
                                                config.activate_num_classes,
                                                config.use_sigmoid_cls)
        self.proposal_generator_test.set_train_local(config, False)

        # Assign and sampler stage two
        self.bbox_assigner_sampler_for_rcnn = BboxAssignSampleForRcnn(config, self.train_batch_size,
                                                                      config.num_bboxes_stage2, True)
        self.bbox_assigner_sampler_for_rcnn1 = BboxAssignSampleForRcnn_1_1(config, self.train_batch_size,
                                                                           config.roi_sample_num, True)
        self.bbox_assigner_sampler_for_rcnn2 = BboxAssignSampleForRcnn_2_1(config, self.train_batch_size,
                                                                           config.roi_sample_num, True)
        self.decode = P.BoundingBoxDecode(max_shape=(config.img_height, config.img_width), means=self.target_means, \
                                          stds=self.target_stds)
        # Roi
        self.roi_init(config)

        # Rcnn
        self.rcnn = Rcnn(config, config.rcnn_in_channels * config.roi_layer['out_size'] * config.roi_layer['out_size'],
                         self.train_batch_size, self.num_classes)
        self.rcnn1 = Rcnn_1(config,
                            config.rcnn_in_channels * config.roi_layer['out_size'] * config.roi_layer['out_size'],
                            self.train_batch_size, self.num_classes)
        self.rcnn2 = Rcnn_2(config,
                            config.rcnn_in_channels * config.roi_layer['out_size'] * config.roi_layer['out_size'],
                            self.train_batch_size, self.num_classes)

        # Op declare
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()

        self.concat = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.concat_2 = P.Concat(axis=2)
        self.reshape = P.Reshape()
        self.select = P.Select()
        self.greater = P.Greater()
        self.transpose = P.Transpose()

        # Improve speed
        self.concat_start = min(self.num_classes - 2, 55)
        self.concat_end = (self.num_classes - 1)

        # Test mode
        self.test_mode_init(config)

        # Init tensor
        self.init_tensor(config)
        self.device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"

        self.R0_node = Transformer(
                d_model=256,
                dropout=0.1,
                nhead=4,
                dim_feedforward=256*4,
                num_encoder_layers=0,
                num_decoder_layers=2,
                normalize_before=False,
                return_intermediate_dec=False,
                rm_self_attn_dec=True,
                rm_first_self_attn=True,
                tgt_seq_length=config["feature_shapes"][-1][0]*config["feature_shapes"][-1][1],
                batch_size=config.batch_size
        )
        self.R0_fc =nn.Dense(256, 1)
        self.query_embed = mindspore.Parameter(Tensor(np.random.rand(1,256), mindspore.float32), requires_grad=True)
        self.positional_encoding = SinePositionalEncoding(num_feats=256 // 2, normalize=True)
        self.R0_loss = nn.BCEWithLogitsLoss()
        self.R0_weight = 0.1
        self.sum_op = P.ReduceSum()

    def roi_init(self, config):
        "initial roi parameters"
        self.roi_align = SingleRoIExtractor(config,
                                            config.roi_layer,
                                            config.roi_align_out_channels,
                                            config.roi_align_featmap_strides,
                                            self.train_batch_size,
                                            config.roi_align_finest_scale)
        self.roi_align.set_train_local(config, True)
        self.roi_align_test = SingleRoIExtractor(config,
                                                 config.roi_layer,
                                                 config.roi_align_out_channels,
                                                 config.roi_align_featmap_strides,
                                                 1,
                                                 config.roi_align_finest_scale)
        self.roi_align_test.set_train_local(config, False)
        #2nd
        self.roi_align1 = SingleRoIExtractor_1(config,
                                               config.roi_layer,
                                               config.roi_align_out_channels,
                                               config.roi_align_featmap_strides,
                                               self.train_batch_size,
                                               config.roi_align_finest_scale)
        self.roi_align1.set_train_local(config, True)
        self.roi_align_test1 = SingleRoIExtractor_1(config,
                                                    config.roi_layer,
                                                    config.roi_align_out_channels,
                                                    config.roi_align_featmap_strides,
                                                    1,
                                                    config.roi_align_finest_scale)
        self.roi_align_test1.set_train_local(config, False)
        #3rd
        self.roi_align2 = SingleRoIExtractor_2(config,
                                               config.roi_layer,
                                               config.roi_align_out_channels,
                                               config.roi_align_featmap_strides,
                                               self.train_batch_size,
                                               config.roi_align_finest_scale)
        self.roi_align2.set_train_local(config, True)
        self.roi_align_test2 = SingleRoIExtractor_2(config,
                                                    config.roi_layer,
                                                    config.roi_align_out_channels,
                                                    config.roi_align_featmap_strides,
                                                    1,
                                                    config.roi_align_finest_scale)
        self.roi_align_test2.set_train_local(config, False)

    def test_mode_init(self, config):
        "initial test mode"
        self.test_batch_size = config.test_batch_size
        self.split = P.Split(axis=0, output_num=self.test_batch_size)
        self.split_shape = P.Split(axis=0, output_num=4)
        self.split_scores = P.Split(axis=1, output_num=self.num_classes)
        self.split_cls = P.Split(axis=0, output_num=self.num_classes-1)
        self.tile = P.Tile()
        self.gather = P.GatherNd()

        self.rpn_max_num = config.rpn_max_num

        self.zeros_for_nms = Tensor(np.zeros((self.rpn_max_num, 3)).astype(self.dtype))
        self.ones_mask = np.ones((self.rpn_max_num, 1)).astype(np.bool)
        self.zeros_mask = np.zeros((self.rpn_max_num, 1)).astype(np.bool)
        self.bbox_mask = Tensor(np.concatenate((self.ones_mask, self.zeros_mask,
                                                self.ones_mask, self.zeros_mask), axis=1))
        self.nms_pad_mask = Tensor(np.concatenate((self.ones_mask, self.ones_mask,
                                                   self.ones_mask, self.ones_mask, self.zeros_mask), axis=1))

        self.test_score_thresh = Tensor(np.ones((self.rpn_max_num, 1)).astype(self.dtype) * config.test_score_thr)
        self.test_score_zeros = Tensor(np.ones((self.rpn_max_num, 1)).astype(self.dtype) * 0)
        self.test_box_zeros = Tensor(np.ones((self.rpn_max_num, 4)).astype(self.dtype) * -1)
        self.test_iou_thr = Tensor(np.ones((self.rpn_max_num, 1)).astype(self.dtype) * config.test_iou_thr)
        self.test_max_per_img = config.test_max_per_img
        self.nms_test = P.NMSWithMask(config.test_iou_thr)
        self.softmax = P.Softmax(axis=1)
        self.logicand = P.LogicalAnd()
        self.oneslike = P.OnesLike()
        self.test_topk = P.TopK(sorted=True)
        self.test_num_proposal = self.test_batch_size * self.rpn_max_num

    def init_tensor(self, config):
        roi_align_index = [np.array(np.ones((config.num_expected_pos_stage2 + config.num_expected_neg_stage2, 1)) * i,
                                    dtype=self.dtype) for i in range(self.train_batch_size)]

        roi_align_index_test = [np.array(np.ones((config.rpn_max_num, 1)) * i, dtype=self.dtype) \
                                for i in range(self.test_batch_size)]

        self.roi_align_index_tensor = Tensor(np.concatenate(roi_align_index))
        self.roi_align_index_test_tensor = Tensor(np.concatenate(roi_align_index_test))

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids):
        "Cascade rcnn method"
        x = self.backbone(img_data)
        x = self.R1_feature(x)

        ####################R0_node#######################
        R0_target = ops.zeros(len(gt_bboxes), mindspore.float32)
        for i in range(gt_bboxes.shape[0]):
            if gt_labels[i][0] != 0:
                R0_target[i] = 1
        R0_target = F.stop_gradient(R0_target)
        x0 = x[-1]
        batch_size = x0.shape[0]
        input_img_h = self.img_h
        input_img_w = self.img_w
        masks = Tensor(np.ones((batch_size, input_img_h, input_img_w)), mindspore.float32)
        for img_id in range(batch_size):
            img_h = self.img_h
            img_w = self.img_w
            masks[img_id, :img_h, :img_w] = 0

        # interpolate masks to have the same spatial shape with x
        nearest_interpolate_op = ops.ResizeNearestNeighbor(x0.shape[-2:])
        masks = nearest_interpolate_op(ops.expand_dims(masks, 1)).squeeze(1).astype(mindspore.bool_)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        R0_feature = self.R0_node(x0, self.query_embed, pos_embed)[0]
        # R0_feature = ops.adaptive_avg_pool2d(x0, (1,1)).squeeze()
        confidence = self.R0_fc(R0_feature)
        confidence_score = ops.sigmoid(ops.stop_gradient(confidence))
        loss_confidence = self.R0_loss(confidence.squeeze(1), R0_target) * self.R0_weight

        selected_id = Tensor(np.zeros((self.train_batch_size)), dtype=mindspore.int32)
        for i in range(self.train_batch_size):
            if confidence_score[i] > 0.5:
                selected_id[i] = 1
        ####################R0_node#######################

        rpn_loss, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss, _ = self.rpn_with_loss(x,
                                                                                           img_metas,
                                                                                           self.anchor_list,
                                                                                           gt_bboxes,
                                                                                           self.gt_labels_stage1,
                                                                                           gt_valids,
                                                                                           selected_id)

        if self.training:
            proposal, proposal_mask = self.proposal_generator(cls_score, bbox_pred, self.anchor_list)
        else:
            proposal, proposal_mask = self.proposal_generator_test(cls_score, bbox_pred, self.anchor_list)

        if self.training:
            gt_labels = self.cast(gt_labels, mstype.int32)
            gt_valids = self.cast(gt_valids, mstype.int32)

        roi_feats, bbox_targets, rcnn_labels, rcnn_mask_squeeze, rois, _ = self.rcnn_stage1(x, gt_bboxes,
                                                                                            gt_labels, gt_valids,
                                                                                            proposal_mask, proposal)
        x_reg_1st, rcnn_loss_1st, _, _, _ = self.rcnn(roi_feats,
                                                      bbox_targets,
                                                      rcnn_labels,
                                                      rcnn_mask_squeeze, selected_id, 1)
        rois_2nd = self.get_rois_decode(rois, x_reg_1st, self.train_batch_size)

        roi_feats_2, bbox_targets_2, rcnn_labels_2, \
        rcnn_mask_squeeze_2, rois_2nd, _ = self.rcnn_stage2(x, gt_bboxes,
                                                            gt_labels,
                                                            gt_valids,
                                                            rcnn_mask_squeeze,
                                                            rois_2nd,
                                                            proposal_mask)
        x_reg_2st, rcnn_loss_2st, _, _, _ = self.rcnn1(roi_feats_2,
                                                       bbox_targets_2,
                                                       rcnn_labels_2,
                                                       rcnn_mask_squeeze_2, selected_id, 2)
        rois_3rd = self.get_rois_decode(rois_2nd, x_reg_2st, self.train_batch_size)
        #3rd
        bboxes_tuple_3 = ()
        deltas_tuple_3 = ()
        labels_tuple_3 = ()
        mask_tuple_3 = ()
        rcnn_mask_3rd = self.reshape(rcnn_mask_squeeze_2, (self.train_batch_size, -1))
        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)

                bboxes_3rd, deltas_3rd, labels_3rd, mask_3rd = self.bbox_assigner_sampler_for_rcnn2(gt_bboxes_i,
                                                                                                    gt_labels_i,
                                                                                                    rcnn_mask_3rd[i],
                                                                                                    rois_3rd[i],
                                                                                                    gt_valids_i)
                bboxes_tuple_3 += (bboxes_3rd,)
                deltas_tuple_3 += (deltas_3rd,)
                labels_tuple_3 += (labels_3rd,)
                mask_tuple_3 += (mask_3rd,)

            bbox_targets_3 = self.concat(deltas_tuple_3)
            rcnn_labels_3 = self.concat(labels_tuple_3)
            bbox_targets_3 = F.stop_gradient(bbox_targets_3)
            rcnn_labels_3 = F.stop_gradient(rcnn_labels_3)
            rcnn_labels_3 = self.cast(rcnn_labels_3, mstype.int32)
        else:
            bboxes_tuple_3 = self.reshape(rois_3rd, (-1, 4))
            mask_tuple_3 += proposal_mask
            bbox_targets_3 = proposal_mask
            rcnn_labels_3 = proposal_mask

        rois_3rd, bboxes_all_3 = self.get_boxes_3rd(bboxes_tuple_3)
        rois_3rd = self.cast(rois_3rd, mstype.float32)
        rois_3rd = F.stop_gradient(rois_3rd)

        if self.training:
            roi_feats_3 = self.roi_align2(rois_3rd,
                                          self.cast(x[0], mstype.float32),
                                          self.cast(x[1], mstype.float32),
                                          self.cast(x[2], mstype.float32),
                                          self.cast(x[3], mstype.float32))
        else:
            roi_feats_3 = self.roi_align_test2(rois_3rd,
                                               self.cast(x[0], mstype.float32),
                                               self.cast(x[1], mstype.float32),
                                               self.cast(x[2], mstype.float32),
                                               self.cast(x[3], mstype.float32))

        roi_feats_3 = self.cast(roi_feats_3, self.ms_type)
        rcnn_masks_3 = self.concat(mask_tuple_3)
        rcnn_masks_3 = F.stop_gradient(rcnn_masks_3)
        rcnn_mask_squeeze_3 = self.squeeze(self.cast(rcnn_masks_3, mstype.bool_))

        _, rcnn_loss_3rd, rcnn_cls_loss_3rd, rcnn_reg_loss_3rd, _ = self.rcnn2(roi_feats_3,
                                                                               bbox_targets_3,
                                                                               rcnn_labels_3,
                                                                               rcnn_mask_squeeze_3, selected_id, 3)

        output = ()
        if self.training:
            output += (rpn_loss, rcnn_loss_1st, rcnn_loss_2st, rcnn_loss_3rd,
                       rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss_3rd, rcnn_reg_loss_3rd, loss_confidence)
        else:
            output = list(self.get_det_bboxes(rcnn_cls_loss_3rd, rcnn_reg_loss_3rd, rcnn_masks_3, bboxes_all_3, img_metas))
            for i in range(self.train_batch_size):
                if selected_id[i] == 0:
                    output[0][i, :, 4] = 0
                    output[1][i, :, 0] = 0

        return output

    def get_boxes_3rd(self, bboxes_tuple_3):
        "get boxes in 3rd stage"
        if self.training:
            if self.train_batch_size > 1:
                bboxes_all_3 = self.concat(bboxes_tuple_3)
            else:
                bboxes_all_3 = bboxes_tuple_3[0]
            rois_3rd = self.concat_1((self.roi_align_index_tensor, bboxes_all_3))
        else:
            if self.test_batch_size > 1:
                bboxes_all_3 = bboxes_tuple_3
            else:
                bboxes_all_3 = bboxes_tuple_3[0]
            if self.device_type == "Ascend":
                bboxes_all_3 = self.cast(bboxes_all_3, mstype.float32)
            rois_3rd = self.concat_1((self.roi_align_index_test_tensor, bboxes_all_3))
        return rois_3rd, bboxes_all_3

    def rcnn_stage1(self, x, gt_bboxes, gt_labels, gt_valids, proposal_mask, proposal):
        "RCNN Stage"
        bboxes_tuple = ()
        deltas_tuple = ()
        labels_tuple = ()
        mask_tuple = ()
        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)

                bboxes, deltas, labels, mask = self.bbox_assigner_sampler_for_rcnn(gt_bboxes_i,
                                                                                   gt_labels_i,
                                                                                   proposal_mask[i],
                                                                                   proposal[i][::, 0:4:1],
                                                                                   gt_valids_i)
                bboxes_tuple += (bboxes,)
                deltas_tuple += (deltas,)
                labels_tuple += (labels,)
                mask_tuple += (mask,)

            bbox_targets = self.concat(deltas_tuple)
            rcnn_labels = self.concat(labels_tuple)
            bbox_targets = F.stop_gradient(bbox_targets)
            rcnn_labels = F.stop_gradient(rcnn_labels)
            rcnn_labels = self.cast(rcnn_labels, mstype.int32)
        else:
            mask_tuple += proposal_mask
            bbox_targets = proposal_mask
            rcnn_labels = proposal_mask
            for p_i in proposal:
                bboxes_tuple += (p_i[::, 0:4:1],)

        if self.training:
            if self.train_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_tensor, bboxes_all))
        else:
            if self.test_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            if self.device_type == "Ascend":
                bboxes_all = self.cast(bboxes_all, mstype.float32)
            rois = self.concat_1((self.roi_align_index_test_tensor, bboxes_all))

        rois = self.cast(rois, mstype.float32)
        rois = F.stop_gradient(rois)

        if self.training:
            roi_feats = self.roi_align(rois,
                                       self.cast(x[0], mstype.float32),
                                       self.cast(x[1], mstype.float32),
                                       self.cast(x[2], mstype.float32),
                                       self.cast(x[3], mstype.float32))
        else:
            roi_feats = self.roi_align_test(rois,
                                            self.cast(x[0], mstype.float32),
                                            self.cast(x[1], mstype.float32),
                                            self.cast(x[2], mstype.float32),
                                            self.cast(x[3], mstype.float32))

        roi_feats = self.cast(roi_feats, self.ms_type)
        rcnn_masks = self.concat(mask_tuple)
        rcnn_masks = F.stop_gradient(rcnn_masks)
        rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))

        return roi_feats, bbox_targets, rcnn_labels, rcnn_mask_squeeze, rois, bboxes_all

    def rcnn_stage2(self, x, gt_bboxes, gt_labels, gt_valids, rcnn_mask_squeeze, rois_2nd, proposal_mask):
        "RCNN Stage2"
        bboxes_tuple_2 = ()
        deltas_tuple_2 = ()
        labels_tuple_2 = ()
        mask_tuple_2 = ()
        rcnn_mask_2nd = self.reshape(rcnn_mask_squeeze, (self.train_batch_size, -1))
        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)


                bboxes_2nd, deltas_2nd, labels_2nd, mask_2nd = self.bbox_assigner_sampler_for_rcnn1(gt_bboxes_i,
                                                                                                    gt_labels_i,
                                                                                                    rcnn_mask_2nd[i],
                                                                                                    rois_2nd[i],
                                                                                                    gt_valids_i)
                bboxes_tuple_2 += (bboxes_2nd,)
                deltas_tuple_2 += (deltas_2nd,)
                labels_tuple_2 += (labels_2nd,)
                mask_tuple_2 += (mask_2nd,)

            bbox_targets_2 = self.concat(deltas_tuple_2)
            rcnn_labels_2 = self.concat(labels_tuple_2)
            bbox_targets_2 = F.stop_gradient(bbox_targets_2)
            rcnn_labels_2 = F.stop_gradient(rcnn_labels_2)
            rcnn_labels_2 = self.cast(rcnn_labels_2, mstype.int32)
        else:
            bboxes_tuple_2 = self.reshape(rois_2nd, (-1, 4))
            mask_tuple_2 += proposal_mask
            bbox_targets_2 = proposal_mask
            rcnn_labels_2 = proposal_mask

        if self.training:
            if self.train_batch_size > 1:
                bboxes_all_2 = self.concat(bboxes_tuple_2)
            else:
                bboxes_all_2 = bboxes_tuple_2[0]
            rois_2nd = self.concat_1((self.roi_align_index_tensor, bboxes_all_2))
        else:
            if self.test_batch_size > 1:
                bboxes_all_2 = bboxes_tuple_2
            else:
                bboxes_all_2 = bboxes_tuple_2[0]
            if self.device_type == "Ascend":
                bboxes_all_2 = self.cast(bboxes_all_2, mstype.float32)
            rois_2nd = self.concat_1((self.roi_align_index_test_tensor, bboxes_all_2))

        rois_2nd = self.cast(rois_2nd, mstype.float32)
        rois_2nd = F.stop_gradient(rois_2nd)

        if self.training:
            roi_feats_2 = self.roi_align1(rois_2nd,
                                          self.cast(x[0], mstype.float32),
                                          self.cast(x[1], mstype.float32),
                                          self.cast(x[2], mstype.float32),
                                          self.cast(x[3], mstype.float32))
        else:
            roi_feats_2 = self.roi_align_test1(rois_2nd,
                                               self.cast(x[0], mstype.float32),
                                               self.cast(x[1], mstype.float32),
                                               self.cast(x[2], mstype.float32),
                                               self.cast(x[3], mstype.float32))

        roi_feats_2 = self.cast(roi_feats_2, self.ms_type)
        rcnn_masks_2 = self.concat(mask_tuple_2)
        rcnn_masks_2 = F.stop_gradient(rcnn_masks_2)
        rcnn_mask_squeeze_2 = self.squeeze(self.cast(rcnn_masks_2, mstype.bool_))
        return roi_feats_2, bbox_targets_2, rcnn_labels_2, rcnn_mask_squeeze_2, rois_2nd, bboxes_all_2

    def get_rois_decode(self, rois, bbox_pred, batch_size):
        boxes = rois[:, 1:5]
        boxes = self.cast(boxes, mstype.float32)
        box_deltas = bbox_pred
        pred_boxes = self.decode(boxes, box_deltas)
        ret_boxes = pred_boxes.view(batch_size, -1, 4)
        return ret_boxes

    def get_det_bboxes(self, cls_logits, reg_logits, mask_logits, rois, img_metas):
        """Get the actual detection box."""
        scores = self.softmax(cls_logits)

        boxes_all = ()
        for i in range(self.num_classes):
            k = i * 4
            reg_logits_i = self.squeeze(reg_logits[::, k:k+4:1])
            out_boxes_i = self.decode(rois, reg_logits_i)
            boxes_all += (out_boxes_i,)

        img_metas_all = self.split(img_metas)
        scores_all = self.split(scores)
        mask_all = self.split(self.cast(mask_logits, mstype.int32))

        boxes_all_with_batchsize = ()
        for i in range(self.test_batch_size):
            scale = self.split_shape(self.squeeze(img_metas_all[i]))
            scale_h = scale[2]
            scale_w = scale[3]
            boxes_tuple = ()
            for j in range(self.num_classes):
                boxes_tmp = self.split(boxes_all[j])
                out_boxes_h = boxes_tmp[i] / scale_h
                out_boxes_w = boxes_tmp[i] / scale_w
                boxes_tuple += (self.select(self.bbox_mask, out_boxes_w, out_boxes_h),)
            boxes_all_with_batchsize += (boxes_tuple,)

        output = self.multiclass_nms(boxes_all_with_batchsize, scores_all, mask_all)

        return output

    def multiclass_nms(self, boxes_all, scores_all, mask_all):
        """Multiscale postprocessing."""
        all_bboxes = ()
        all_labels = ()
        all_masks = ()

        for i in range(self.test_batch_size):
            bboxes = boxes_all[i]
            scores = scores_all[i]
            masks = self.cast(mask_all[i], mstype.bool_)

            res_boxes_tuple = ()
            res_labels_tuple = ()
            res_masks_tuple = ()

            for j in range(self.num_classes - 1):
                k = j + 1
                _cls_scores = scores[::, k:k + 1:1]
                _bboxes = self.squeeze(bboxes[k])
                _mask_o = self.reshape(masks, (self.rpn_max_num, 1))

                cls_mask = self.greater(_cls_scores, self.test_score_thresh)
                _mask = self.logicand(_mask_o, cls_mask)

                _reg_mask = self.cast(self.tile(self.cast(_mask, mstype.int32), (1, 4)), mstype.bool_)

                _bboxes = self.select(_reg_mask, _bboxes, self.test_box_zeros)
                _cls_scores = self.select(_mask, _cls_scores, self.test_score_zeros)
                __cls_scores = self.squeeze(_cls_scores)
                scores_sorted, topk_inds = self.test_topk(__cls_scores, self.rpn_max_num)
                topk_inds = self.reshape(topk_inds, (self.rpn_max_num, 1))
                scores_sorted = self.reshape(scores_sorted, (self.rpn_max_num, 1))
                _bboxes_sorted = self.gather(_bboxes, topk_inds)
                _mask_sorted = self.gather(_mask, topk_inds)

                scores_sorted = self.tile(scores_sorted, (1, 4))
                cls_dets = self.concat_1((_bboxes_sorted, scores_sorted))
                cls_dets = P.Slice()(cls_dets, (0, 0), (self.rpn_max_num, 5))

                cls_dets, _index, _mask_nms = self.nms_test(cls_dets)
                _index = self.reshape(_index, (self.rpn_max_num, 1))
                _mask_nms = self.reshape(_mask_nms, (self.rpn_max_num, 1))

                _mask_n = self.gather(_mask_sorted, _index)

                _mask_n = self.logicand(_mask_n, _mask_nms)
                cls_labels = self.oneslike(_index) * j
                res_boxes_tuple += (cls_dets,)
                res_labels_tuple += (cls_labels,)
                res_masks_tuple += (_mask_n,)

            res_boxes_start = self.concat(res_boxes_tuple[:self.concat_start])
            res_labels_start = self.concat(res_labels_tuple[:self.concat_start])
            res_masks_start = self.concat(res_masks_tuple[:self.concat_start])

            res_boxes_end = self.concat(res_boxes_tuple[self.concat_start:self.concat_end])
            res_labels_end = self.concat(res_labels_tuple[self.concat_start:self.concat_end])
            res_masks_end = self.concat(res_masks_tuple[self.concat_start:self.concat_end])

            res_boxes = self.concat((res_boxes_start, res_boxes_end))
            res_labels = self.concat((res_labels_start, res_labels_end))
            res_masks = self.concat((res_masks_start, res_masks_end))

            reshape_size = (self.num_classes - 1) * self.rpn_max_num
            res_boxes = self.reshape(res_boxes, (1, reshape_size, 5))
            res_labels = self.reshape(res_labels, (1, reshape_size, 1))
            res_masks = self.reshape(res_masks, (1, reshape_size, 1))

            all_bboxes += (res_boxes,)
            all_labels += (res_labels,)
            all_masks += (res_masks,)

        all_bboxes = self.concat(all_bboxes)
        all_labels = self.concat(all_labels)
        all_masks = self.concat(all_masks)
        return all_bboxes, all_labels, all_masks

    def get_anchors(self, featmap_sizes):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = ()
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors += (Tensor(anchors.astype(self.dtype)),)

        return multi_level_anchors

class CascadeRcnn_Infer(nn.Cell):
    def __init__(self, config):
        super(CascadeRcnn_Infer, self).__init__()
        self.network = Cascade_Rcnn_Resnet101(config)
        self.network.set_train(False)

    def construct(self, img_data, img_metas):
        output = self.network(img_data, img_metas, None, None, None)
        return output
