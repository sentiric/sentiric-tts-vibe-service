# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch VibeVoice model."""

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from transformers.activations import get_activation
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_phonemizer_available,
    logging,
    requires_backends,
)
from .configuration_vibevoice import VibeVoiceConfig

if is_phonemizer_available():
    from phonemizer.backend import EspeakBackend
    from phonemizer.punctuation import Punctuation
    from phonemizer.separator import Separator

logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "VibeVoiceConfig"
_CHECKPOINT_FOR_DOC = "microsoft/VibeVoice-1.5B"


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class VibeVoiceResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=get_padding(kernel_size, dilation[i]),
                    )
                )
                for i in range(len(dilation))
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for i in range(len(dilation))
            ]
        )

        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class VibeVoiceGenerator(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(config.hidden_size, config.upsample_initial_channel, 7, 1, padding=3))

        # --- YENİ EKLENEN SATIR ---
        self.cond = Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)
        # --- DEĞİŞİKLİK BİTTİ ---
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        config.upsample_initial_channel // (2**i),
                        config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(VibeVoiceResidualBlock(ch, k, d, config.leaky_relu_slope))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.leaky_relu_slope = config.leaky_relu_slope

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            # --- YENİ EKLENEN SATIR ---
            g = self.cond(g) # g'nin boyutunu 256'dan 512'ye çıkar
            # --- DEĞİŞİKLİK BİTTİ --            
            x = x + g
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class VibeVoiceTextEncoder(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.max_position_embeddings = config.max_position_embeddings

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embedding.weight, 0.0, config.hidden_size**-0.5)

        self.encoder = VibeVoiceEncoder(config)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x * math.sqrt(self.hidden_size)

        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min
            extended_attention_mask = extended_attention_mask.squeeze(1)
        else:
            extended_attention_mask = None

        encoder_outputs = self.encoder(x, extended_attention_mask)
        return encoder_outputs


class VibeVoiceEncoder(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [VibeVoiceEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        for layer_module in self.layers:
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        return BaseModelOutput(last_hidden_state=hidden_states)


class VibeVoiceEncoderLayer(nn.Module):
    def __init__(self, config: VibeVoiceConfig) -> None:
        super().__init__()
        self.attention = VibeVoiceAttention(config)
        self.intermediate = VibeVoiceIntermediate(config)
        self.output = VibeVoiceOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
        )
        attention_output = self_attention_outputs[0]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return (layer_output,)


class VibeVoiceAttention(nn.Module):
    def __init__(self, config: VibeVoiceConfig) -> None:
        super().__init__()
        self.attention = VibeVoiceSelfAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.attention(hidden_states, attention_mask)
        return self_outputs


class VibeVoiceSelfAttention(nn.Module):
    def __init__(self, config: VibeVoiceConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.relative_positional_encoding = VibeVoiceRelativePositionalEncoding(config)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = self.relative_positional_encoding(query_layer, attention_scores)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(1)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        attention_output = self.output(context_layer)
        return (attention_output,)


class VibeVoiceRelativePositionalEncoding(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.relative_positional_bias = nn.Embedding(2 * config.max_position_embeddings, self.attention_head_size)

    def forward(self, query: torch.Tensor, attention_scores: torch.Tensor):
        batch_size, num_heads, seq_len, head_dim = query.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=query.device)
        relative_position_ids = position_ids[None, :] - position_ids[:, None]
        relative_position_ids = relative_position_ids.clone() + (self.relative_positional_bias.num_embeddings // 2)
        relative_positional_bias = self.relative_positional_bias(relative_position_ids)
        rel_logits = torch.matmul(query, relative_positional_bias.transpose(-1, -2))
        return attention_scores + rel_logits


class VibeVoiceIntermediate(nn.Module):
    def __init__(self, config: VibeVoiceConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(config.hidden_size, config.intermediate_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.intermediate_size, config.hidden_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class VibeVoiceOutput(nn.Module):
    def __init__(self, config: VibeVoiceConfig) -> None:
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + input_tensor
        return hidden_states


class VibeVoicePosteriorEncoder(nn.Module):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.out_channels = config.hidden_size * 2
        self.hidden_size = config.hidden_size
        self.pre = nn.Conv1d(config.spec_channels, config.hidden_size, 1)
        self.enc = VibeVoiceWaveNet(config.hidden_size, 1, 5, 16, gin_channels=config.hidden_size)
        self.proj = nn.Conv1d(config.hidden_size, self.out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        if g is not None:
            g = F.normalize(g)
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.hidden_size, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class VibeVoiceWaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(0.05)

        if gin_channels != 0:
            self.cond_layer = Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels]).to(x.device)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class VibeVoiceStochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.log_flow = VibeVoiceLog()
        self.flows = nn.ModuleList()
        self.flows.append(VibeVoiceElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(VibeVoiceConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(VibeVoiceFlip())
        self.pre = Conv1d(in_channels, filter_channels, 1)
        self.proj = Conv1d(filter_channels, filter_channels, 1)
        self.convs = VibeVoiceDDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask
        if not reverse:
            return None
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, 1, 1)
            logw = z0
            return logw


class VibeVoiceLog(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class VibeVoiceElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class VibeVoiceConvFlow(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2
        self.pre = Conv1d(self.half_channels, filter_channels, 1)
        self.convs = VibeVoiceDDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = Conv1d(filter_channels, self.half_channels * (num_bins * 3), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask
        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)
        
        unnormalized_widths = h[..., : self.num_bins]
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = h[..., 2 * self.num_bins :]
        
        # Çağrıyı `unconstrained_rational_quadratic_spline` olarak düzeltiyoruz
        x1_reshaped, logabsdet = unconstrained_rational_quadratic_spline(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tail_bound=self.tail_bound,
        )
        x1 = x1_reshaped.view(b, c, t) # Boyutu orijinal haline getir
        
        y = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return y, logdet
        else:
            return y


class VibeVoiceFlip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class VibeVoiceDDSConv(nn.Module):
    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
            )
            self.convs_1x1.append(Conv1d(channels, channels, 1))
            self.norms_1.append(VibeVoiceLayerNorm(channels))
            self.norms_2.append(VibeVoiceLayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class VibeVoiceLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


@dataclass
class VibeVoiceModelOutput(ModelOutput):
    waveform: torch.FloatTensor = None
    spectrogram: torch.FloatTensor = None
    log_durations: torch.FloatTensor = None
    total_nll: torch.FloatTensor = None
    kl_loss: torch.FloatTensor = None


class VibeVoicePreTrainedModel(PreTrainedModel):
    config_class = VibeVoiceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = [r"discriminator"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # --- BU SATIRI DEĞİŞTİR ---
                # ESKİ HALİ: module.bias.data.zero()
                # YENİ HALİ:
                module.bias.data.zero_()
                # --- DEĞİŞİKLİK BİTTİ ---
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # --- BU SATIRI DEĞİŞTİR ---
                # ESKİ HALİ: module.weight.data[module.padding_idx].zero()
                # YENİ HALİ:
                module.weight.data[module.padding_idx].zero_()
                # --- DEĞİŞİKLİK BİTTİ ---
        elif isinstance(module, nn.LayerNorm):
            # --- BU SATIRI DEĞİŞTİR ---
            # ESKİ HALİ: module.bias.data.zero()
            # YENİ HALİ:
            module.bias.data.zero_()
            # --- DEĞİŞİKLİK BİTTİ ---
            module.weight.data.fill_(1.0)

# --- BU EKSİK BLOĞU BURAYA EKLE ---
VIBEVOICE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VibeVoiceConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIBEVOICE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        speaker_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            The ID of the speaker to use for synthesis.
        language_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            The ID of the language to use for synthesis. If the model is monolingual, this should not be used.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

TEXT_SYNTHESIS_EXAMPLE = r"""
    Example of text synthesis:

    ```python
    >>> from transformers import AutoProcessor, VibeVoiceModel
    >>> from IPython.display import Audio

    >>> processor = AutoProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
    >>> model = VibeVoiceModel.from_pretrained("microsoft/VibeVoice-1.5B")

    >>> # English
    >>> inputs = processor("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model.generate(**inputs, language="en")
    >>> Audio(outputs.waveform.numpy(), rate=model.config.sampling_rate)
    ```

    Example of voice cloning:

    ```python
    >>> import torch
    >>> from transformers import AutoProcessor, VibeVoiceModel
    >>> from IPython.display import Audio
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
    >>> model = VibeVoiceModel.from_pretrained("microsoft/VibeVoice-1.5B")

    >>> # English
    >>> inputs = processor("Hello, my dog is cute", return_tensors="pt")
    >>> # load and process audio
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> audio = dataset[0]["audio"]
    >>> speaker_embeddings = processor(audios=audio["array"], return_tensors="pt")
    >>> speaker_embeddings = model.get_speaker_embedding(speaker_embeddings)
    >>> speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings)
    >>> outputs = model.generate(**inputs, speaker_embeddings=speaker_embeddings, language="en")
    >>> Audio(outputs.waveform.numpy(), rate=model.config.sampling_rate)
    ```
"""
# --- EKLEME BİTTİ --

# --- NİHAİ DÜZELTME: Sorunlu spline fonksiyonları yerine, daha sağlam,
# standart bir implementasyon (VITS'in kullandığı) eklendi. ---
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError(f"{tails} tails is not implemented.")

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, logabsdet

def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    
    # --- BU KONTROL BLOĞUNU EKLE ---
    if inputs.numel() == 0:
        return inputs, torch.zeros_like(inputs)
    # --- DEĞİŞİKLİK BİTTİ ---
        
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("min_bin_width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("min_bin_height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = torch.searchsorted(cumheights, inputs.unsqueeze(-1)).squeeze(-1) - 1
    else:
        bin_idx = torch.searchsorted(cumwidths, inputs.unsqueeze(-1)).squeeze(-1) - 1

    input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_derivatives = derivatives.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    bin_idx_plus_one = torch.clamp(bin_idx + 1, 0, num_bins - 1)
    input_derivatives_plus_one = derivatives.gather(-1, bin_idx_plus_one.unsqueeze(-1)).squeeze(-1)
    

    input_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    if inverse:
        a = (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) + input_heights * (
            input_delta - input_derivatives
        )
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        discriminant = b.pow(2) - 4 * a * c
        if torch.min(discriminant) < 0:
            raise ValueError("Invalid discriminant")
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_less_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_less_theta)
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * root * (1 - root)
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_less_theta = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_less_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_less_theta)
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_less_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet
# --- NİHAİ DÜZELTME BİTTİ ---

@add_start_docstrings("VibeVoice model for text-to-speech.", VIBEVOICE_START_DOCSTRING)
class VibeVoiceModel(VibeVoicePreTrainedModel):
    def __init__(self, config: VibeVoiceConfig):
        super().__init__(config)
        self.config = config
        self.text_encoder = VibeVoiceTextEncoder(config)
        self.duration_predictor = VibeVoiceStochasticDurationPredictor(
            config.hidden_size, 192, 3, 0.5, 4, gin_channels=config.speaker_embedding_size
        )
        self.posterior_encoder = VibeVoicePosteriorEncoder(config)
        self.generator = VibeVoiceGenerator(config)
        self.speaker_embedding = nn.Embedding(config.speaker_embedding_size, config.speaker_embedding_size)
        self.language_embedding = nn.Embedding(config.num_languages, config.hidden_size)
        self.speaker_projection = nn.Linear(config.speaker_embedding_size, config.hidden_size)
        self.language_projection = nn.Linear(config.hidden_size, config.speaker_embedding_size)
        self.post_init()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        tokenizer,
        attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> "VibeVoiceModelOutput":
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if language is not None:
            requires_backends(self, "phonemizer")
            phonemizer = EspeakBackend(language=language, with_stress=True)
            separator = Separator(phone="-", word=" ", syllable="|")
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            phonemes = phonemizer.phonemize([text], separator=separator, strip=True)
            processed_inputs = tokenizer(phonemes, return_tensors="pt")
            input_ids = processed_inputs["input_ids"].to(self.device)
            attention_mask = processed_inputs["attention_mask"].to(self.device)
        
        if speaker_embeddings is None:
            speaker_embeddings = self.speaker_embedding.weight.mean(0).unsqueeze(0)

        speaker_embeddings_norm = F.normalize(speaker_embeddings)
        text_encoder_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        hidden_states = text_encoder_outputs.last_hidden_state.transpose(1, 2)
        text_mask = attention_mask.unsqueeze(1)

        if self.config.use_spk_conditioned_encoder:
            speaker_proj = self.speaker_projection(speaker_embeddings_norm).unsqueeze(-1)
            hidden_states = hidden_states + speaker_proj
        
        g = speaker_embeddings_norm.unsqueeze(-1)

        log_durations = self.duration_predictor(
            hidden_states, text_mask, g=g, reverse=True, noise_scale=0.667
        )
        
        durations = torch.exp(log_durations) * text_mask
        
        # --- NİHAİ DÜZELTME BURADA ---
        # "Sıfır Süre" ve "Çok Kısa Girdi" sorunlarını çözmek için
        # sürelere küçük bir taban ekleyip bir çarpanla ölçekliyoruz.
        # 1.2 değeri, sesin biraz daha yavaş ve anlaşılır olmasını sağlar.
        durations = (durations + 0.1) * 1.2
        # --- DÜZELTME BİTTİ ---

        durations = torch.ceil(durations).long()
        
        audio_len = torch.sum(durations, dim=-1)[0]
        
        # Eğer hala ses uzunluğu çok kısaysa, bir hata fırlatmak yerine minimum bir uzunluk verelim.
        if audio_len < 1:
            audio_len = torch.tensor(1, device=self.device)
        
        attn = torch.zeros(1, 1, hidden_states.size(-1), audio_len.item(), device=self.device)
        
        start_frame = 0
        for i in range(hidden_states.size(-1)):
            duration = durations[0, 0, i]
            if duration > 0:
                end_frame = start_frame + duration
                # attn'in boyutunu aşmadığından emin olalım
                if end_frame > audio_len.item():
                    end_frame = audio_len.item()
                attn[0, 0, i, start_frame:end_frame] = 1
                start_frame = end_frame
        
        hidden_states = torch.matmul(hidden_states, attn.squeeze(1))
        
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        
        waveform = self.generator(hidden_states, g=g)

        return VibeVoiceModelOutput(waveform=waveform.squeeze(1))