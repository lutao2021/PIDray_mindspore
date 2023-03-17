import copy
from typing import Optional, List

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer, XavierUniform
from .MultiHeadAttention import MultiheadAttention


class Transformer(nn.Cell):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 rm_self_attn_dec=True, rm_first_self_attn=True,
                 tgt_seq_length=None, batch_size=2
                 ):
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        if num_decoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, tgt_seq_length=tgt_seq_length)
            encoder_norm = nn.LayerNorm((d_model,), begin_norm_axis=2, begin_params_axis=2) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                tgt_seq_length=tgt_seq_length, batch_size=batch_size)
        decoder_norm = nn.LayerNorm((d_model,), begin_norm_axis=2, begin_params_axis=2)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.rm_self_attn_dec = rm_self_attn_dec
        self.rm_first_self_attn = rm_first_self_attn

        if self.rm_self_attn_dec or self.rm_first_self_attn:
            self.rm_self_attn_dec_func()

        # self.debug_mode = False
        # self.set_debug_mode(self.debug_mode)

    def rm_self_attn_dec_func(self):
        total_modifie_layer_num = 0
        rm_list = []
        for idx, layer in enumerate(self.decoder.layers):
            if idx == 0 and not self.rm_first_self_attn:
                continue
            if idx != 0 and not self.rm_self_attn_dec:
                continue

            layer.omit_selfattn = True
            del layer.self_attn
            del layer.dropout1
            del layer.norm1

            total_modifie_layer_num += 1
            rm_list.append(idx)
        # remove some self-attention layer
        # print("rm {} layer: {}".format(total_modifie_layer_num, rm_list))

    def set_debug_mode(self, status):
        print("set debug mode to {}!!!".format(status))
        self.debug_mode = status
        if hasattr(self, 'encoder'):
            for idx, layer in enumerate(self.encoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)
        if hasattr(self, 'decoder'):
            for idx, layer in enumerate(self.decoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)

    def _reset_parameters(self):
        for p in self.trainable_params():
            if p.ndim > 1:
                p.set_data(initializer(XavierUniform(), p.shape, mindspore.float32))

    def construct(self, src, query_embed, pos_embed, mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = ops.transpose(src.view(src.shape[0], src.shape[1], -1), (2, 0, 1))
        pos_embed = ops.transpose(pos_embed.view(pos_embed.shape[0], pos_embed.shape[1], -1), (2, 0, 1))
        query_embed = ops.repeat_elements(ops.expand_dims(query_embed, 1), bs, 1)
        if mask is not None:
            mask = mask.view(mask.shape[0], -1)

        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src

        tgt = ops.ZerosLike()(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs.squeeze(), memory[:h * w].transpose(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Cell):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Cell):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def construct(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return ops.stack(intermediate)

        return ops.expand_dims(output, 0)


class TransformerEncoderLayer(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_seq_length=None):
        super().__init__()
        self.self_attn = nn.transformer.MultiHeadAttention(batch_size=2, src_seq_length=1, tgt_seq_length=tgt_seq_length,
                                                           hidden_size=d_model, num_heads=nhead,
                                                           hidden_dropout_rate=dropout, attention_dropout_rate=dropout,
                                                           compute_dtype=mindspore.float32)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm((d_model,), begin_norm_axis=1, begin_params_axis=1)
        self.norm2 = nn.LayerNorm((d_model,), begin_norm_axis=1, begin_params_axis=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, src, attention_mask=src_mask)
        src2 = src2.transpose(1, 0)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2, attention_mask=src_mask)[0]

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def construct(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_seq_length=None, batch_size=2):
        super().__init__()
        self.tgt_seq_length = tgt_seq_length
        self.batch_size = batch_size
        self.self_attn = nn.transformer.MultiHeadAttention(batch_size=batch_size, src_seq_length=1, tgt_seq_length=tgt_seq_length,
                                                           hidden_size=d_model, num_heads=nhead,
                                                           hidden_dropout_rate=dropout, attention_dropout_rate=dropout,
                                                           compute_dtype=mindspore.float32)
        self.multihead_attn = nn.transformer.MultiHeadAttention(batch_size=batch_size, src_seq_length=1, tgt_seq_length=tgt_seq_length,
                                                           hidden_size=d_model, num_heads=nhead,
                                                           hidden_dropout_rate=dropout, attention_dropout_rate=dropout,
                                                           compute_dtype=mindspore.float32)
        # self.multihead_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, attn_drop=dropout, proj_drop=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm((d_model,), begin_norm_axis=2, begin_params_axis=2)
        self.norm2 = nn.LayerNorm((d_model,), begin_norm_axis=2, begin_params_axis=2)
        self.norm3 = nn.LayerNorm((d_model,), begin_norm_axis=2, begin_params_axis=2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None
        self.omit_selfattn = False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        if not self.omit_selfattn:
            tgt2, sim_mat_1 = self.self_attn(q, k, tgt, attention_mask=tgt_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        if memory_mask is None:
            memory_mask = Tensor(np.ones((self.batch_size, 1, self.tgt_seq_length)), dtype=mindspore.float32)
        tgt2, sim_mat_2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).transpose(1,0,2),
                                              self.with_pos_embed(memory, pos).transpose(1,0,2),
                                              memory.transpose(1,0,2), attention_mask=memory_mask)
        # tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).transpose(1,0,2),
        #                                       self.with_pos_embed(memory, pos).transpose(1,0,2),
        #                                       memory.transpose(1,0,2), attn_mask=None)
        # tgt2 = memory.mean(0).view(memory.shape[1], 1, memory.shape[2])

        tgt2 = tgt2.transpose(1,0,2)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=tgt_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt2, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attention_mask=memory_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def construct(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])

