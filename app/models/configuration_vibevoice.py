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
""" VibeVoice model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

VIBEVOICE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/VibeVoice-1.5B": "https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/config.json",
}


class VibeVoiceConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceModel`]. It is used to instantiate a
    VibeVoice model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VibeVoice
    [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 395):
            Vocabulary size of the VibeVoice model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`VibeVoiceModel`].
        hidden_size (`int`, *optional*, defaults to 192):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 768):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        speaker_embedding_size (`int`, *optional*, defaults to 256):
            The number of speakers in the speaker embedding.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate of the audio waveform.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The initial channel size of the upsampler.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[6, 5, 5, 2]`):
            The upsample rates of the upsampler.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[12, 10, 10, 4]`):
            The kernel sizes of the upsampler.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            The kernel sizes of the residual blocks.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            The dilation sizes of the residual blocks.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The slope of the leaky relu activation function.
        spec_segment_size (`int`, *optional*, defaults to 32):
            The segment size of the spectrogram.
        spec_channels (`int`, *optional*, defaults to 1025):
            The number of channels in the spectrogram.
        num_languages (`int`, *optional*, defaults to 10):
            The number of languages in the language embedding.
        use_spk_conditioned_encoder (`bool`, *optional*, defaults to `True`):
            Whether or not to use speaker conditioned encoder.
        use_noise_scaled_mas (`bool`, *optional*, defaults to `True`):
            Whether or not to use noise scaled MAS.
        mas_noise_scale_initial (`float`, *optional*, defaults to 0.01):
            The initial noise scale of the MAS.
        mas_noise_scale_delta (`float`, *optional*, defaults to 2e-06):
            The noise scale delta of the MAS.
        vocoder_model_name (`str`, *optional*):
            The name of the vocoder model to use.

    Example:

    ```python
    >>> from transformers import VibeVoiceConfig, VibeVoiceModel

    >>> # Initializing a VibeVoice microsoft/VibeVoice-1.5B style configuration
    >>> configuration = VibeVoiceConfig()

    >>> # Initializing a model from the microsoft/VibeVoice-1.5B style configuration
    >>> model = VibeVoiceModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vibevoice"

    def __init__(
        self,
        vocab_size=395,
        hidden_size=192,
        num_hidden_layers=6,
        num_attention_heads=2,
        intermediate_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        max_position_embeddings=2048,
        speaker_embedding_size=256,
        sampling_rate=24000,
        upsample_initial_channel=512,
        upsample_rates=[6, 5, 5, 2],
        upsample_kernel_sizes=[12, 10, 10, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        leaky_relu_slope=0.1,
        spec_segment_size=32,
        spec_channels=1025,
        num_languages=10,
        use_spk_conditioned_encoder=True,
        use_noise_scaled_mas=True,
        mas_noise_scale_initial=0.01,
        mas_noise_scale_delta=2e-06,
        vocoder_model_name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.speaker_embedding_size = speaker_embedding_size
        self.sampling_rate = sampling_rate
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.leaky_relu_slope = leaky_relu_slope
        self.spec_segment_size = spec_segment_size
        self.spec_channels = spec_channels
        self.num_languages = num_languages
        self.use_spk_conditioned_encoder = use_spk_conditioned_encoder
        self.use_noise_scaled_mas = use_noise_scaled_mas
        self.mas_noise_scale_initial = mas_noise_scale_initial
        self.mas_noise_scale_delta = mas_noise_scale_delta
        self.vocoder_model_name = vocoder_model_name