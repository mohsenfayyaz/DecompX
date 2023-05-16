# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .decompx_utils import DecompXConfig, DecompXOutput

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.configuration_bert import BertConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model

def output_builder(input_vector, output_mode):
    if output_mode is None:
        return None
    elif output_mode == "vector":
        return (input_vector,)
    elif output_mode == "norm":
        return (torch.norm(input_vector, dim=-1),)
    elif output_mode == "both":
        return ((torch.norm(input_vector, dim=-1), input_vector),)
    elif output_mode == "distance_based":
        recomposed_vectors = torch.sum(input_vector, dim=-2, keepdim=True)
        importance_matrix = -torch.nn.functional.pairwise_distance(input_vector, recomposed_vectors, p=1)
        norm_y = torch.norm(recomposed_vectors, dim=-1, p=1)
        maxed = torch.maximum(torch.zeros(1, device=norm_y.device), norm_y + importance_matrix)
        return (maxed / (torch.sum(maxed, dim=-2, keepdim=True) + 1e-12),)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_for_decomposed(self, x):
        # x: (B, N, N, H*V)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # x: (B, N, N, H, V)
        x = x.view(new_x_shape)
        # x: (B, H, N, N, V)
        return x.permute(0, 3, 1, 2, 4)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attribution_vectors: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
            decompx_ready: Optional[bool] = None,  # added by Fayyaz / Modarressi
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        decomposed_value_layer = None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            if attribution_vectors is not None:
                decomposed_value_layer = torch.einsum("bijd,vd->bijv", attribution_vectors, self.value.weight)
                decomposed_value_layer = self.transpose_for_scores_for_decomposed(decomposed_value_layer)
            

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # added by Fayyaz / Modarressi
        # -------------------------------
        if decompx_ready:
            outputs = (context_layer, attention_probs, value_layer, decomposed_value_layer)
            return outputs
        # -------------------------------

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor,
                decompx_ready=False):  # added by Fayyaz / Modarressi
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        pre_ln_states = hidden_states + input_tensor  # added by Fayyaz / Modarressi
        post_ln_states = self.LayerNorm(pre_ln_states)  # added by Fayyaz / Modarressi
        # added by Fayyaz / Modarressi
        if decompx_ready:
            return post_ln_states, pre_ln_states
        else:
            return post_ln_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attribution_vectors: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
            decompx_ready: Optional[bool] = None,  # added by Fayyaz / Modarressi
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attribution_vectors,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            decompx_ready=decompx_ready,  # added by Fayyaz / Modarressi
        )
        attention_output = self.output(
            self_outputs[0],
            hidden_states,
            decompx_ready=decompx_ready,  # added by Goro Kobayashi (Edited by Fayyaz / Modarressi)
        )

        # Added by Fayyaz / Modarressi
        # -------------------------------
        if decompx_ready:
            _, attention_probs, value_layer, decomposed_value_layer = self_outputs
            attention_output, pre_ln_states = attention_output
            outputs = (attention_output, attention_probs,) + (value_layer, decomposed_value_layer, pre_ln_states)  # add attentions and norms if we output them
            return outputs
        # -------------------------------

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor, decompx_ready: Optional[bool] = False) -> torch.Tensor:
        pre_act_hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(pre_act_hidden_states)
        if decompx_ready:
            return hidden_states, pre_act_hidden_states
        return hidden_states, None


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, decompx_ready: Optional[bool] = False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # return hidden_states
        # Added by Fayyaz / Modarressi
        # -------------------------------
        pre_ln_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(pre_ln_states)
        if decompx_ready:
            return hidden_states, pre_ln_states
        return hidden_states, None
        # -------------------------------


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.similarity_fn = torch.nn.CosineSimilarity(dim=-1)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def bias_decomposer(self, bias, attribution_vectors, bias_decomp_type="absdot"):
        # Decomposes the input bias based on similarity to the attribution vectors
        # Args:
        #   bias: a bias vector (all_head_size)
        #   attribution_vectors: the attribution vectors from token j to i (b, i, j, all_head_size) :: (batch, seq_length, seq_length, all_head_size) 

        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bskd,d->bsk", attribution_vectors, bias))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(attribution_vectors, bias, dim=-1))
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(attribution_vectors, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(attribution_vectors.shape[:-1], device=attribution_vectors.device)
            weights[:,:,0] = 1.0
        elif bias_decomp_type == "dot":
            weights = torch.einsum("bskd,d->bsk", attribution_vectors, bias)
        elif bias_decomp_type == "biastoken":
            attrib_shape = attribution_vectors.shape
            if attrib_shape[1] == attrib_shape[2]:
                attribution_vectors = torch.concat([attribution_vectors, torch.zeros((attrib_shape[0], attrib_shape[1], 1, attrib_shape[3]), device=attribution_vectors.device)], dim=-2)
            attribution_vectors[:,:,-1] = attribution_vectors[:,:,-1] + bias
            return attribution_vectors

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.matmul(weights.unsqueeze(dim=-1), bias.unsqueeze(dim=0))
        return attribution_vectors + weighted_bias


    def ln_decomposer(self, attribution_vectors, pre_ln_states, gamma, beta, eps, include_biases=True, bias_decomp_type="absdot"):
        mean = pre_ln_states.mean(-1, keepdim=True)  # (batch, seq_len, 1) m(y=Σy_j)
        var = (pre_ln_states - mean).pow(2).mean(-1, keepdim=True).unsqueeze(dim=2)  # (batch, seq_len, 1, 1)  s(y)

        each_mean = attribution_vectors.mean(-1, keepdim=True)  # (batch, seq_len, seq_len, 1) m(y_j)

        normalized_layer = torch.div(attribution_vectors - each_mean,
                                         (var + eps) ** (1 / 2))  # (batch, seq_len, seq_len, all_head_size)

        post_ln_layer = torch.einsum('bskd,d->bskd', normalized_layer,
                                         gamma)  # (batch, seq_len, seq_len, all_head_size)
        
        if include_biases:
            return self.bias_decomposer(beta, post_ln_layer, bias_decomp_type=bias_decomp_type)
        else:
            return post_ln_layer 


    def gelu_linear_approximation(self, intermediate_hidden_states, intermediate_output):
        def phi(x):
            return (1 + torch.erf(x / math.sqrt(2))) / 2.
        
        def normal_pdf(x):
            return torch.exp(-(x**2) / 2) / math.sqrt(2. * math.pi)

        def gelu_deriv(x):
            return phi(x)+x*normal_pdf(x)
        
        m = gelu_deriv(intermediate_hidden_states)
        b = intermediate_output - m * intermediate_hidden_states
        return m, b


    def gelu_decomposition(self, attribution_vectors, intermediate_hidden_states, intermediate_output, bias_decomp_type):
        m, b = self.gelu_linear_approximation(intermediate_hidden_states, intermediate_output)
        mx = attribution_vectors * m.unsqueeze(dim=-2)

        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bskl,bsl->bsk", mx, b))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(mx, b))
            weights = (torch.norm(mx, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(mx, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(mx, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(mx.shape[:-1], device=mx.device)
            weights[:,:,0] = 1.0

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.einsum("bsl,bsk->bskl", b, weights)
        return mx + weighted_bias

    def gelu_zo_decomposition(self, attribution_vectors, intermediate_hidden_states, intermediate_output):
        m = intermediate_output / (intermediate_hidden_states + 1e-12)
        mx = attribution_vectors * m.unsqueeze(dim=-2)
        return mx
    
    def ffn_decomposer(self, attribution_vectors, intermediate_hidden_states, intermediate_output, include_biases=True, approximation_type="GeLU_LA", bias_decomp_type="absdot"):
        post_first_layer = torch.einsum("ld,bskd->bskl", self.intermediate.dense.weight, attribution_vectors)
        if include_biases:
            post_first_layer = self.bias_decomposer(self.intermediate.dense.bias, post_first_layer, bias_decomp_type=bias_decomp_type)

        if approximation_type == "ReLU":
            mask_for_gelu_approx = (intermediate_hidden_states > 0)
            post_act_first_layer = torch.einsum("bskl, bsl->bskl", post_first_layer, mask_for_gelu_approx)
            post_act_first_layer = post_first_layer * mask_for_gelu_approx.unsqueeze(dim=-2)
        elif approximation_type == "GeLU_LA":
            post_act_first_layer = self.gelu_decomposition(post_first_layer, intermediate_hidden_states, intermediate_output, bias_decomp_type=bias_decomp_type)
        elif approximation_type == "GeLU_ZO":
            post_act_first_layer = self.gelu_zo_decomposition(post_first_layer, intermediate_hidden_states, intermediate_output)

        post_second_layer = torch.einsum("bskl, dl->bskd", post_act_first_layer, self.output.dense.weight)
        if include_biases:
            post_second_layer = self.bias_decomposer(self.output.dense.bias, post_second_layer, bias_decomp_type=bias_decomp_type)

        return post_second_layer

    def ffn_decomposer_fast(self, attribution_vectors, intermediate_hidden_states, intermediate_output, include_biases=True, approximation_type="GeLU_LA", bias_decomp_type="absdot"):
        if approximation_type == "ReLU":
            theta = (intermediate_hidden_states > 0)
        elif approximation_type == "GeLU_ZO":
            theta = intermediate_output / (intermediate_hidden_states + 1e-12)
        
        scaled_W1 = torch.einsum("bsl,ld->bsld", theta, self.intermediate.dense.weight)
        W_equiv = torch.einsum("bsld, zl->bszd", scaled_W1, self.output.dense.weight)

        post_ffn_layer = torch.einsum("bszd,bskd->bskz", W_equiv, attribution_vectors)

        if include_biases:
            scaled_b1 = torch.einsum("bsl,l->bsl", theta, self.intermediate.dense.bias)
            b_equiv = torch.einsum("bsl, dl->bsd", scaled_b1, self.output.dense.weight)
            b_equiv = b_equiv + self.output.dense.bias

            if bias_decomp_type == "absdot":
                weights = torch.abs(torch.einsum("bskd,bsd->bsk", post_ffn_layer, b_equiv))
            elif bias_decomp_type == "abssim":
                weights = torch.abs(torch.nn.functional.cosine_similarity(post_ffn_layer, b_equiv))
                weights = (torch.norm(post_ffn_layer, dim=-1) != 0) * weights
            elif bias_decomp_type == "norm":
                weights = torch.norm(post_ffn_layer, dim=-1)
            elif bias_decomp_type == "equal":
                weights = (torch.norm(post_ffn_layer, dim=-1) != 0) * 1.0

            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
            weighted_bias = torch.einsum("bsd,bsk->bskd", b_equiv, weights)

            post_ffn_layer = post_ffn_layer + weighted_bias

        return post_ffn_layer

    def forward(
            self,
            hidden_states: torch.Tensor,
            attribution_vectors: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
            decompx_config: Optional[DecompXConfig] = None, # added by Fayyaz / Modarressi
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # self_attention_outputs = self.attention(
        #     hidden_states,
        #     attention_mask,
        #     head_mask,
        #     output_attentions=output_attentions,
        #     past_key_value=self_attn_past_key_value,
        # )
        decompx_ready = decompx_config is not None
        self_attention_outputs = self.attention(
            hidden_states,
            attribution_vectors,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            decompx_ready=decompx_ready,
        )  # changed by Goro Kobayashi
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        # )

        # Added by Fayyaz / Modarressi
        # -------------------------------
        bias_decomp_type = "biastoken" if decompx_config.include_bias_token else decompx_config.bias_decomp_type
        intermediate_output, pre_act_hidden_states = self.intermediate(attention_output, decompx_ready=decompx_ready)
        layer_output, pre_ln2_states = self.output(intermediate_output, attention_output, decompx_ready=decompx_ready)
        if decompx_ready:
            attention_probs, value_layer, decomposed_value_layer, pre_ln_states = outputs

            headmixing_weight = self.attention.output.dense.weight.view(self.all_head_size, self.num_attention_heads,
                                      self.attention_head_size)

            if decomposed_value_layer is None or decompx_config.aggregation != "vector":
                transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer, headmixing_weight)  # V * W^o  (z=(qk)v)
                # Make weighted vectors αf(x) from transformed vectors (transformed_layer)
                # and attention weights (attentions):
                # (batch, num_heads, seq_length, seq_length, all_head_size)
                weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs,
                                          transformed_layer)  # attention_probs(Q*K^t) * V * W^o
                # Sum each weighted vectors αf(x) over all heads:
                # (batch, seq_length, seq_length, all_head_size)
                summed_weighted_layer = weighted_layer.sum(dim=1)  # sum over heads

                # Make residual matrix (batch, seq_length, seq_length, all_head_size)
                hidden_shape = hidden_states.size()  # (batch, seq_length, all_head_size)
                device = hidden_states.device
                residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device),
                                        hidden_states)  # diagonal representations (hidden states)

                # Make matrix of summed weighted vector + residual vectors
                residual_weighted_layer = summed_weighted_layer + residual
                accumulated_bias = self.attention.output.dense.bias
            else:
                transformed_layer = torch.einsum('bhsqv,dhv->bhsqd', decomposed_value_layer, headmixing_weight)

                weighted_layer = torch.einsum('bhks,bhsqd->bhkqd', attention_probs,
                                          transformed_layer)  # attention_probs(Q*K^t) * V * W^o

                summed_weighted_layer = weighted_layer.sum(dim=1)  # sum over heads

                residual_weighted_layer = summed_weighted_layer + attribution_vectors
                accumulated_bias = torch.matmul(self.attention.output.dense.weight, self.attention.self.value.bias) + self.attention.output.dense.bias

            if decompx_config.include_biases:
                residual_weighted_layer = self.bias_decomposer(accumulated_bias, residual_weighted_layer, bias_decomp_type)

            if decompx_config.include_LN1:
                post_ln_layer = self.ln_decomposer(
                    attribution_vectors=residual_weighted_layer,
                    pre_ln_states=pre_ln_states,
                    gamma=self.attention.output.LayerNorm.weight.data,
                    beta=self.attention.output.LayerNorm.bias.data,
                    eps=self.attention.output.LayerNorm.eps,
                    include_biases=decompx_config.include_biases,
                    bias_decomp_type=bias_decomp_type
                )
            else:
                post_ln_layer = residual_weighted_layer

            if decompx_config.include_FFN:
                post_ffn_layer = self.ffn_decomposer_fast if decompx_config.FFN_fast_mode else self.ffn_decomposer(
                    attribution_vectors=post_ln_layer,
                    intermediate_hidden_states=pre_act_hidden_states,
                    intermediate_output=intermediate_output,
                    approximation_type=decompx_config.FFN_approx_type,
                    include_biases=decompx_config.include_biases,
                    bias_decomp_type=bias_decomp_type
                )
                pre_ln2_layer = post_ln_layer + post_ffn_layer
            else:
                pre_ln2_layer = post_ln_layer
                post_ffn_layer = None

            if decompx_config.include_LN2:
                post_ln2_layer = self.ln_decomposer(
                    attribution_vectors=pre_ln2_layer,
                    pre_ln_states=pre_ln2_states,
                    gamma=self.output.LayerNorm.weight.data,
                    beta=self.output.LayerNorm.bias.data,
                    eps=self.output.LayerNorm.eps,
                    include_biases=decompx_config.include_biases,
                    bias_decomp_type=bias_decomp_type
                )
            else:
                post_ln2_layer = pre_ln2_layer

            new_outputs = DecompXOutput(
                attention=output_builder(summed_weighted_layer, decompx_config.output_attention),
                res1=output_builder(residual_weighted_layer, decompx_config.output_res1),
                LN1=output_builder(post_ln_layer, decompx_config.output_res2),
                FFN=output_builder(post_ffn_layer, decompx_config.output_FFN),
                res2=output_builder(pre_ln2_layer, decompx_config.output_res2),
                encoder=output_builder(post_ln2_layer, "both")
            )
            return (layer_output,) + (new_outputs,)
        # -------------------------------
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
            decompx_config: Optional[DecompXConfig] = None,  # added by Fayyaz / Modarressi
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        aggregated_encoder_norms = None # added by Fayyaz / Modarressi
        aggregated_encoder_vectors = None # added by Fayyaz / Modarressi

        # -- added by Fayyaz / Modarressi
        if decompx_config and decompx_config.output_all_layers:
            all_decompx_outputs = DecompXOutput(
                attention=() if decompx_config.output_attention else None,
                res1=() if decompx_config.output_res1 else None,
                LN1=() if decompx_config.output_LN1 else None,
                FFN=() if decompx_config.output_LN1 else None,
                res2=() if decompx_config.output_res2 else None,
                encoder=() if decompx_config.output_encoder else None,
                aggregated=() if decompx_config.output_aggregated and decompx_config.aggregation else None,
            )
        else:
            all_decompx_outputs = None
        # -- added by Fayyaz / Modarressi

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    aggregated_encoder_vectors,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    decompx_config # added by Fayyaz / Modarressi
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            
            # added by Fayyaz / Modarressi
            if decompx_config:
                decompx_output = layer_outputs[1]
                if decompx_config.aggregation == "rollout":
                    if decompx_config.include_classifier_w_pooler:
                        raise Exception("Classifier and pooler could be included in vector aggregation mode")

                    encoder_norms = decompx_output.encoder[0][0]

                    if aggregated_encoder_norms is None:
                        aggregated_encoder_norms = encoder_norms * torch.exp(attention_mask).view((-1, attention_mask.shape[-1], 1))
                    else:
                        aggregated_encoder_norms = torch.einsum("ijk,ikm->ijm", encoder_norms, aggregated_encoder_norms)
                        
                    if decompx_config.output_aggregated == "norm":
                        decompx_output.aggregated = (aggregated_encoder_norms,)
                    elif decompx_config.output_aggregated is not None:
                        raise Exception("Rollout aggregated values are only available in norms. Set output_aggregated to 'norm'.")


                elif decompx_config.aggregation == "vector":
                    aggregated_encoder_vectors = decompx_output.encoder[0][1]

                    if decompx_config.include_classifier_w_pooler:
                        decompx_output.aggregated = (aggregated_encoder_vectors,)
                    else:
                        decompx_output.aggregated = output_builder(aggregated_encoder_vectors, decompx_config.output_aggregated)

                decompx_output.encoder = output_builder(decompx_output.encoder[0][1], decompx_config.output_encoder)

                if decompx_config.output_all_layers:
                    all_decompx_outputs.attention = all_decompx_outputs.attention + decompx_output.attention if decompx_config.output_attention else None
                    all_decompx_outputs.res1 = all_decompx_outputs.res1 + decompx_output.res1 if decompx_config.output_res1 else None
                    all_decompx_outputs.LN1 = all_decompx_outputs.LN1 + decompx_output.LN1 if decompx_config.output_LN1 else None
                    all_decompx_outputs.FFN = all_decompx_outputs.FFN + decompx_output.FFN if decompx_config.output_FFN else None
                    all_decompx_outputs.res2 = all_decompx_outputs.res2 + decompx_output.res2 if decompx_config.output_res2 else None
                    all_decompx_outputs.encoder = all_decompx_outputs.encoder + decompx_output.encoder if decompx_config.output_encoder else None

                    if decompx_config.include_classifier_w_pooler and decompx_config.aggregation == "vector":
                        all_decompx_outputs.aggregated = all_decompx_outputs.aggregated + output_builder(aggregated_encoder_vectors, decompx_config.output_aggregated) if decompx_config.output_aggregated else None
                    else:
                        all_decompx_outputs.aggregated = all_decompx_outputs.aggregated + decompx_output.aggregated if decompx_config.output_aggregated else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                    decompx_output if decompx_config else None,
                    all_decompx_outputs
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, decompx_ready=False) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pre_pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pre_pooled_output)
        if decompx_ready:
            return pooled_output, pre_pooled_output
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def bias_decomposer(self, bias, attribution_vectors, bias_decomp_type="absdot"):
        # Decomposes the input bias based on similarity to the attribution vectors
        # Args:
        #   bias: a bias vector (all_head_size)
        #   attribution_vectors: the attribution vectors from token j to i (b, i, j, all_head_size) :: (batch, seq_length, seq_length, all_head_size)

        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bkd,d->bk", attribution_vectors, bias))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(attribution_vectors, bias, dim=-1))
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(attribution_vectors, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(attribution_vectors.shape[:-1], device=attribution_vectors.device)
            weights[:,0] = 1.0
        elif bias_decomp_type == "dot":
            weights = torch.einsum("bkd,d->bk", attribution_vectors, bias)
        elif bias_decomp_type == "biastoken":
            attribution_vectors[:,-1] = attribution_vectors[:,-1] + bias
            return attribution_vectors
        
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

        weighted_bias = torch.matmul(weights.unsqueeze(dim=-1), bias.unsqueeze(dim=0))
        return attribution_vectors + weighted_bias

    def tanh_linear_approximation(self, pre_act_pooled, post_act_pooled):
        def tanh_deriv(x):
            return 1 - torch.tanh(x)**2.0
        
        m = tanh_deriv(pre_act_pooled)
        b = post_act_pooled - m * pre_act_pooled
        return m, b

    def tanh_la_decomposition(self, attribution_vectors, pre_act_pooled, post_act_pooled, bias_decomp_type):
        m, b = self.tanh_linear_approximation(pre_act_pooled, post_act_pooled)
        mx = attribution_vectors * m.unsqueeze(dim=-2)

        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bkd,bd->bk", mx, b))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(mx, b, dim=-1))
            weights = (torch.norm(mx, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(mx, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(mx, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(mx.shape[:-1], device=mx.device)
            weights[:,0] = 1.0
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.einsum("bd,bk->bkd", b, weights)
        return mx + weighted_bias

    def tanh_zo_decomposition(self, attribution_vectors, pre_act_pooled, post_act_pooled):
        m = post_act_pooled / (pre_act_pooled + 1e-12)
        mx = attribution_vectors * m.unsqueeze(dim=-2)
        return mx
    
    def ffn_decomposer(self, attribution_vectors, pre_act_pooled, post_act_pooled, include_biases=True, bias_decomp_type="absdot", tanh_approx_type="LA"):
        post_pool = torch.einsum("ld,bsd->bsl", self.pooler.dense.weight, attribution_vectors)
        if include_biases:
            post_pool = self.bias_decomposer(self.pooler.dense.bias, post_pool, bias_decomp_type=bias_decomp_type)

        if tanh_approx_type == "LA":
            post_act_pool = self.tanh_la_decomposition(post_pool, pre_act_pooled, post_act_pooled, bias_decomp_type=bias_decomp_type)
        else:
            post_act_pool = self.tanh_zo_decomposition(post_pool, pre_act_pooled, post_act_pooled)

        return post_act_pool

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            decompx_config: Optional[DecompXConfig] = None,  # added by Fayyaz / Modarressi
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            decompx_config=decompx_config, # added by Fayyaz / Modarressi
        )
        sequence_output = encoder_outputs[0]
        decompx_ready = decompx_config is not None
        pooled_output = self.pooler(sequence_output, decompx_ready=decompx_ready) if self.pooler is not None else None

        if decompx_ready:
            pre_act_pooled = pooled_output[1]
            pooled_output = pooled_output[0]

            if decompx_config.include_classifier_w_pooler:
                decompx_idx = -2 if decompx_config.output_all_layers else -1
                aggregated_attribution_vectors = encoder_outputs[decompx_idx].aggregated[0]

                encoder_outputs[decompx_idx].aggregated = output_builder(aggregated_attribution_vectors, decompx_config.output_aggregated)

                pooler_decomposed = self.ffn_decomposer(
                    attribution_vectors=aggregated_attribution_vectors[:, 0], 
                    pre_act_pooled=pre_act_pooled, 
                    post_act_pooled=pooled_output, 
                    include_biases=decompx_config.include_biases,
                    bias_decomp_type="biastoken" if decompx_config.include_bias_token else decompx_config.bias_decomp_type,
                    tanh_approx_type=decompx_config.tanh_approx_type
                )

                encoder_outputs[decompx_idx].pooler = pooler_decomposed

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            next_sentence_label: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning.""", BERT_START_DOCSTRING
)
class BertLMHeadModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.Tensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be
                in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100`
                are ignored (masked), the loss is only computed for the tokens with labels n `[0, ...,
                config.vocab_size]`
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        >>> config = BertConfig.from_pretrained("bert-base-cased")
        >>> config.is_decoder = True
        >>> model = BertLMHeadModel.from_pretrained("bert-base-cased", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def bias_decomposer(self, bias, attribution_vectors, bias_decomp_type="absdot"):
        # Decomposes the input bias based on similarity to the attribution vectors
        # Args:
        #   bias: a bias vector (all_head_size)
        #   attribution_vectors: the attribution vectors from token j to i (b, i, j, all_head_size) :: (batch, seq_length, seq_length, all_head_size) 
        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bkd,d->bk", attribution_vectors, bias))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(attribution_vectors, bias, dim=-1))
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(attribution_vectors, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(attribution_vectors.shape[:-1], device=attribution_vectors.device)
            weights[:,0] = 1.0
        elif bias_decomp_type == "dot":
            weights = torch.einsum("bkd,d->bk", attribution_vectors, bias)
        elif bias_decomp_type == "biastoken":
            attribution_vectors[:,-1] = attribution_vectors[:,-1] + bias
            return attribution_vectors

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.matmul(weights.unsqueeze(dim=-1), bias.unsqueeze(dim=0))
        return attribution_vectors + weighted_bias

    def biastoken_decomposer(self, biastoken, attribution_vectors, bias_decomp_type="absdot"):
        # Decomposes the input bias based on similarity to the attribution vectors
        # Args:
        #   bias: a bias vector (all_head_size)
        #   attribution_vectors: the attribution vectors from token j to i (b, i, j, all_head_size) :: (batch, seq_length, seq_length, all_head_size) 
        if bias_decomp_type == "absdot":
            weights = torch.abs(torch.einsum("bkd,bd->bk", attribution_vectors, biastoken))
        elif bias_decomp_type == "abssim":
            weights = torch.abs(torch.nn.functional.cosine_similarity(attribution_vectors, biastoken, dim=-1))
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * weights
        elif bias_decomp_type == "norm":
            weights = torch.norm(attribution_vectors, dim=-1)
        elif bias_decomp_type == "equal":
            weights = (torch.norm(attribution_vectors, dim=-1) != 0) * 1.0
        elif bias_decomp_type == "cls":
            weights = torch.zeros(attribution_vectors.shape[:-1], device=attribution_vectors.device)
            weights[:,0] = 1.0
        elif bias_decomp_type == "dot":
            weights = torch.einsum("bkd,d->bk", attribution_vectors, biastoken)

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weighted_bias = torch.matmul(weights.unsqueeze(dim=-1), biastoken.unsqueeze(dim=1))
        return attribution_vectors + weighted_bias

    def ffn_decomposer(self, attribution_vectors, include_biases=True, bias_decomp_type="absdot"):
        post_classifier = torch.einsum("ld,bkd->bkl", self.classifier.weight, attribution_vectors)
        if include_biases:
            post_classifier = self.bias_decomposer(self.classifier.bias, post_classifier, bias_decomp_type=bias_decomp_type)

        return post_classifier

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            decompx_config: Optional[DecompXConfig] = None,  # added by Fayyaz / Modarressi
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            decompx_config=decompx_config
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if decompx_config and decompx_config.include_classifier_w_pooler:
            decompx_idx = -2 if decompx_config.output_all_layers else -1
            aggregated_attribution_vectors = outputs[decompx_idx].pooler

            outputs[decompx_idx].pooler = output_builder(aggregated_attribution_vectors, decompx_config.output_pooler)

            classifier_decomposed = self.ffn_decomposer(
                attribution_vectors=aggregated_attribution_vectors, 
                include_biases=decompx_config.include_biases,
                bias_decomp_type="biastoken" if decompx_config.include_bias_token else decompx_config.bias_decomp_type
            )
            
            if decompx_config.include_bias_token and decompx_config.bias_decomp_type is not None:
                bias_token = classifier_decomposed[:,-1,:].detach().clone()
                classifier_decomposed = classifier_decomposed[:,:-1,:]
                classifier_decomposed = self.biastoken_decomposer(
                    bias_token, 
                    classifier_decomposed, 
                    bias_decomp_type=decompx_config.bias_decomp_type
                )
                

            outputs[decompx_idx].classifier = classifier_decomposed if decompx_config.output_classifier else None

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output  # (loss), logits, (hidden_states), (attentions)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
