# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Wav2Vec2 model."""

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2FeedForward,
    Wav2Vec2EncoderLayer,
    Wav2Vec2EncoderLayerStableLayerNorm,
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForXVector,
    AMSoftmaxLoss,
    TDNNLayer,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

_HIDDEN_STATES_START_POSITION = 2

#Wav2Vec2 configuration file
class AdaptiveWav2Vec2Config(Wav2Vec2Config):
    def __init__(self, 
                 *args,
                 use_adapter_attn=True,
                 use_adapter_ffn=True,
                 use_encoder_adapter=False,
                 encoder_adapter_act=nn.GELU,

                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_adapter_attn = use_adapter_attn
        self.use_adapter_ffn = use_adapter_ffn
        self.use_encoder_adapter = use_encoder_adapter
        self.encoder_adapter_act = encoder_adapter_act
        
# Wav2Vec2AdapterLayer
class Wav2Vec2AdapterLayer(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.linear_down = nn.Linear(config.hidden_size, config.adapter_attn_dim[layer])
        self.act = ACT2FN[config.encoder_adapter_act] if config.encoder_adapter_act else None
        self.linear_up = nn.Linear(config.adapter_attn_dim[layer], config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.act(self.linear_down(hidden_states)) \
                            if self.act else self.linear_down(hidden_states)
        hidden_states = self.linear_up(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states    

# Wav2Vec2 Encoder layer
class AdaptiveWav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config: AdaptiveWav2Vec2Config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            config=config,
        )

        self.dropout = nn.Dropout(config.hidden_dropout)
        if config.use_adapter_attn:
            self.adapter_layer_attn = Wav2Vec2AdapterLayer(config)
        else:
            self.adapter_layer_attn = None
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_adapter_ffn:
            self.adapter_layer_ffn = Wav2Vec2AdapterLayer(config)
        else:
            self.adapter_layer_ffn = None
        
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, 
                hidden_states, 
                attention_mask=None, 
                output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)

        if self.adapter_layer_attn is not None:
            hidden_states = self.adapter_layer_attn(hidden_states)

        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)

        if self.adapter_layer_ffn is not None:
            hidden_states = hidden_states + self.adapter_layer_ff(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

# Wav2Vec2 Encoder
class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        attention_mask = self._update_full_mask(
            attention_mask,
            hidden_states,
        )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask


