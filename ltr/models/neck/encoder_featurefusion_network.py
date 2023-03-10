# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ltr.models.backbone.convolutional_block import build_DwConv_Block


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # add pos to src
        if self.divide_norm:
            # print("encoder divide by norm")
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
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
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderDwconvLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

        self.dwconv_block_template = build_DwConv_Block(in_planes=d_model, out_planes=d_model)
        self.dwconv_block_search = build_DwConv_Block(in_planes=d_model, out_planes=d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,src_temp, src_search,
                     src_mask: Optional[Tensor]=None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos_src: Optional[Tensor] = None):

        src_temp = self.dwconv_block_search(src_temp)
        src_search = self.dwconv_block_template(src_search)

        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)

        src = torch.cat([src_temp, src_search], dim=0)

        q = k = self.with_pos_embed(src, pos_src)  # add pos to src
        if self.divide_norm:
            # print("encoder divide by norm")
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoderFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False, need_dwconv=True):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm31 = nn.LayerNorm(d_model)
        self.norm32 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout31 = nn.Dropout(dropout)
        self.dropout32 = nn.Dropout(dropout)

        self.need_dwconv = need_dwconv
        
        if need_dwconv:
            self.dwconv_block_template = build_DwConv_Block(in_planes=d_model, out_planes=d_model)
            self.dwconv_block_search = build_DwConv_Block(in_planes=d_model, out_planes=d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None
                ):
                     
        if self.need_dwconv:
            src1 = self.dwconv_block_search(src1)
            src2 = self.dwconv_block_template(src2)

        src1 = src1.flatten(2).permute(2, 0, 1)
        src2 = src2.flatten(2).permute(2, 0, 1)

        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                               key_padding_mask=src1_key_padding_mask)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        q2 = k2 = self.with_pos_embed(src2, pos_src2)
        src22 = self.self_attn2(q2, k2, value=src2, attn_mask=src2_mask,
                               key_padding_mask=src2_key_padding_mask)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)

        src = torch.cat([src1, src2], dim=0)
        feat_mask = torch.cat([src1_key_padding_mask, src2_key_padding_mask], dim=1)
        pos_embed = torch.cat([pos_src1, pos_src2], dim=0)

        q = k = self.with_pos_embed(src, pos_embed)  # add pos to src
        src31 = self.self_attn3(q, k, src, attn_mask=None,
                              key_padding_mask=feat_mask)[0]
        src = src + self.dropout31(src31)
        src = self.norm31(src)

        src32 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout32(src32)
        src = self.norm32(src)
        return src

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_intermediate=False):
        if return_intermediate:
            output_list = []
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is None:
                    output_list.append(output)
            if self.norm is not None:
                output = self.norm(output)
                output_list.append(output)
            return output_list
        else:
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

            if self.norm is not None:
                output = self.norm(output)

            return output

class TransformerDWconvEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src_temp, src_search,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_intermediate=False):
        if return_intermediate:
            output_list = []
            # output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is None:
                    output_list.append(output)
            if self.norm is not None:
                output = self.norm(output)
                output_list.append(output)
            return output_list
        else:
            # output = src
            B, C, H_t, W_t = src_temp.shape
            _, _, H_s, W_s = src_search.shape

            for layer in self.layers:
                output = layer(src_temp, src_search, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos_src=pos)
                src_temp, src_search = torch.split(output, [H_t*W_t, H_s*W_s], dim=0)
                src_temp, src_search = src_temp.permute(1, 2, 0).reshape(B, C, H_t, W_t), src_search.permute(1, 2, 0).reshape(B, C, H_s, W_s)

            if self.norm is not None:
                output = self.norm(output)

            return output

class TransformerFusionEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src_temp, src_search,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

            # output = src
            B, C, H_t, W_t = src_temp.shape
            _, _, H_s, W_s = src_search.shape

            for layer in self.layers:
                output = layer(src_temp, src_search, 
                               src1_mask=src1_mask, src2_mask=src2_mask, 
                               src1_key_padding_mask=src1_key_padding_mask, src2_key_padding_mask=src2_key_padding_mask,
                               pos_src1=pos_src1, pos_src2=pos_src2)
                src_temp, src_search = torch.split(output, [H_t*W_t, H_s*W_s], dim=0)
                src_temp, src_search = src_temp.permute(1, 2, 0).reshape(B, C, H_t, W_t), src_search.permute(1, 2, 0).reshape(B, C, H_s, W_s)

            if self.norm is not None:
                output = self.norm(output)

            return output


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.dim = d_model

        self.dwconv_block_template = build_DwConv_Block(in_planes=self.dim, out_planes=self.dim)
        self.dwconv_block_search = build_DwConv_Block(in_planes=self.dim, out_planes=self.dim)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search, mode="encoder"):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        
        B, C, H_t, W_t = src_temp.shape
        _, _, H_s, W_s = src_search.shape

        src_temp = self.dwconv_block_search(src_temp)
        src_search = self.dwconv_block_template(src_search)

        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        feat = torch.cat([src_temp, src_search], dim=0)
        mask = torch.cat([mask_temp, mask_search], dim=1)
        pos_embed = torch.cat([pos_temp, pos_search], dim=0)

        assert mode in ["all", "encoder"]
        if self.encoder is None:
            memory = feat
        else:
            memory = self.encoder(feat, src_key_padding_mask=mask, pos=pos_embed)

        memory_temp, memory_search = torch.split(memory, [H_t*W_t, H_s*W_s], dim=0)
        memory_temp, memory_search = memory_temp.permute(1, 2, 0).reshape(B, C, H_t, W_t), memory_search.permute(1, 2, 0).reshape(B, C, H_s, W_s)

        return memory_temp, memory_search

class TransformerFusion(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False, need_dwconv=True):
        super().__init__()
        self.dim = d_model

        # self.dwconv_block_template = build_DwConv_Block(in_planes=self.dim, out_planes=self.dim)
        # self.dwconv_block_search = build_DwConv_Block(in_planes=self.dim, out_planes=self.dim)

        encoder_layer = TransformerEncoderFusionLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before, divide_norm=divide_norm, need_dwconv=need_dwconv)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerFusionEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search, mode="encoder"):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        
        B, C, H_t, W_t = src_temp.shape
        _, _, H_s, W_s = src_search.shape

        # src_temp = self.dwconv_block_search(src_temp)
        # src_search = self.dwconv_block_template(src_search)

        # src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        # src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        # feat = torch.cat([src_temp, src_search], dim=0)
        # mask = torch.cat([mask_temp, mask_search], dim=1)
        # pos_embed = torch.cat([pos_temp, pos_search], dim=0)

        assert mode in ["all", "encoder"]
        if self.encoder is None:
            raise ValueError
        else:
            memory = self.encoder(src_temp, src_search, src1_key_padding_mask=mask_temp, src2_key_padding_mask=mask_search, pos_src1=pos_temp, pos_src2=pos_search)

        memory_temp, memory_search = torch.split(memory, [H_t*W_t, H_s*W_s], dim=0)
        memory_temp, memory_search = memory_temp.permute(1, 2, 0).reshape(B, C, H_t, W_t), memory_search.permute(1, 2, 0).reshape(B, C, H_s, W_s)

        return memory_temp, memory_search

class TransformerDWconv(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.dim = d_model

        encoder_layer = TransformerEncoderDwconvLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerDWconvEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search, mode="encoder"):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        
        B, C, H_t, W_t = src_temp.shape
        _, _, H_s, W_s = src_search.shape

        # src_temp = self.dwconv_block_search(src_temp)
        # src_search = self.dwconv_block_template(src_search)

        # src_temp = src_temp.flatten(2).permute(2, 0, 1)
        # src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        # feat = torch.cat([src_temp, src_search], dim=0)
        mask = torch.cat([mask_temp, mask_search], dim=1)
        pos_embed = torch.cat([pos_temp, pos_search], dim=0)

        assert mode in ["all", "encoder"]
        if self.encoder is None:
            raise ValueError
        else:
            memory = self.encoder(src_temp, src_search, src_key_padding_mask=mask, pos=pos_embed)

        memory_temp, memory_search = torch.split(memory, [H_t*W_t, H_s*W_s], dim=0)
        memory_temp, memory_search = memory_temp.permute(1, 2, 0).reshape(B, C, H_t, W_t), memory_search.permute(1, 2, 0).reshape(B, C, H_s, W_s)

        return memory_temp, memory_search

class Transformer_withoutdwconv(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.dim = d_model

        # self.dwconv_block_template = build_DwConv_Block(in_planes=self.dim, out_planes=self.dim)
        # self.dwconv_block_search = build_DwConv_Block(in_planes=self.dim, out_planes=self.dim)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search, mode="encoder"):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        
        B, C, H_t, W_t = src_temp.shape
        _, _, H_s, W_s = src_search.shape

        # src_temp = self.dwconv_block_search(src_temp)
        # src_search = self.dwconv_block_template(src_search)

        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        feat = torch.cat([src_temp, src_search], dim=0)
        mask = torch.cat([mask_temp, mask_search], dim=1)
        pos_embed = torch.cat([pos_temp, pos_search], dim=0)

        assert mode in ["all", "encoder"]
        if self.encoder is None:
            memory = feat
        else:
            memory = self.encoder(feat, src_key_padding_mask=mask, pos=pos_embed)

        memory_temp, memory_search = torch.split(memory, [H_t*W_t, H_s*W_s], dim=0)
        memory_temp, memory_search = memory_temp.permute(1, 2, 0).reshape(B, C, H_t, W_t), memory_search.permute(1, 2, 0).reshape(B, C, H_s, W_s)

        return memory_temp, memory_search


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_encoder_featurefusion_network(params, settings, output_layer):
    return Transformer(
        d_model=params['embed_dim'] * 2 ** (output_layer),
        dropout=0.1,
        nhead=params['num_heads'][-1], 
        dim_feedforward=(params['embed_dim'] * 2 ** (output_layer))*4,
        num_encoder_layers=params["depths"][-1]
    )

def build_encoder_featurefusion_withoutdwconv_network(params, settings, output_layer):
    return Transformer_withoutdwconv(
        d_model=params['embed_dim'] * 2 ** (output_layer),
        dropout=0.1,
        nhead=params['num_heads'][-1], 
        dim_feedforward=(params['embed_dim'] * 2 ** (output_layer))*4,
        num_encoder_layers=params["depths"][-1]
    )

def build_dwconv_encoder_featurefusion_network(params, settings, output_layer):
    return TransformerDWconv(
        d_model=params['embed_dim'] * 2 ** (output_layer),
        dropout=0.1,
        nhead=params['num_heads'][-1], 
        dim_feedforward=(params['embed_dim'] * 2 ** (output_layer))*4,
        num_encoder_layers=params["depths"][-1]
    )


def build_encoder_featurefusion_enhance_network(params, settings, output_layer, need_dwconv=True):
    return TransformerFusion(
        d_model=params['embed_dim'] * 2 ** (output_layer),
        dropout=0.1,
        nhead=params['num_heads'][-1], 
        dim_feedforward=(params['embed_dim'] * 2 ** (output_layer))*4,
        num_encoder_layers=params["depths"][-1],
        need_dwconv=need_dwconv
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
