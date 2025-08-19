#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append(".") # Adds higher directory to python modules path.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.generation.utils import GenerateOutput


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        # print(self.model.get_vision_tower())
        # exit()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        override_mask = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if hasattr(self, "hook_manager"):
            logits = self.hook_manager("output", ret = logits)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "image_sizes": kwargs.get("image_sizes", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
                "image_sizes": kwargs.get("image_sizes", None),
            }
        )
        return model_inputs

    def prepare_inputs_for_generation_custom(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_custom", None),
                "image_sizes": kwargs.get("image_sizes", None),
            }
        )
        return model_inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)



### These wrapped llava classes may involve redundant code from irrelevant experiments. Just ignore them if found so.
class LcdLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = FlexLlamaModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        
        return_cd_scores=None,
        return_dict: Optional[bool] = None,
        cd_layer = None,
        cd_embs = None,
        last_n_tokens = 1,
        vis_start = None,
        layer_alpha = None,
        allow_modify = False,
        override_mask=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        if override_mask is not None:
            attention_mask[:, :override_mask.shape[1]] = override_mask
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if allow_modify:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cd_layer=cd_layer,
                cd_embs=cd_embs,
                cd_alpha = cd_alpha,
                last_n_tokens = last_n_tokens,
                vis_start= vis_start,
                layer_alpha=layer_alpha,
                allow_modify = allow_modify,

            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cd_layer=cd_layer,
                cd_embs=cd_embs,
                cd_alpha = cd_alpha,
                last_n_tokens = last_n_tokens,
                vis_start= vis_start,
                layer_alpha=layer_alpha,

            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if hasattr(self, "hook_manager"):
            logits = self.hook_manager("output", ret = logits)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

from transformers.models.llama.modeling_llama import *
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask
class FlexLlamaModel(LlavaLlamaModel):
    def __init__(self, config):
        super().__init__(config)
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cd_layer = None,
        cd_embs = None,
        cd_alpha = 1,
        last_n_tokens = 1,
        vis_start = None,
        layer_alpha = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # print('mask', attention_mask)
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_unnormed_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if cd_layer == None:
            for _, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                    all_unnormed_states += (hidden_states, )

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        **kwargs,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:

                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        else:
            for i_layer, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                    all_unnormed_states += (hidden_states, )


                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        **kwargs,
                    )
                else:
                    
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )
                    if i_layer + 1 in cd_layer:
                        layer_cd_embs = cd_embs[i_layer+1]
                        if layer_alpha is not None:
                            cd_alpha = layer_alpha[i_layer]
                        if last_n_tokens == -1:
                            layer_outputs[0][:, :vis_start, :] = (1 + cd_alpha) * layer_outputs[0][:, :vis_start, :] - cd_alpha * layer_cd_embs[:, :vis_start, :]
                            layer_outputs[0][:, vis_start+576:, :] = (1 + cd_alpha) * layer_outputs[0][:, vis_start+576:, :] - cd_alpha * layer_cd_embs[:, vis_start:, :]
                        elif last_n_tokens == 0:
                            layer_outputs[0][:, 0, :] = (1 + cd_alpha) * layer_outputs[0][:, 0, :] - cd_alpha * layer_cd_embs[:, 0, :]
                        else:
                            layer_outputs[0][:, -last_n_tokens:, :] = (1 + cd_alpha) * layer_outputs[0][:, -last_n_tokens:, :] - cd_alpha * layer_cd_embs


                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        unnormed_states = hidden_states.clone()
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_unnormed_states += (unnormed_states,)

        next_cache = None
        if use_cache:
            if not cd_layer:
                next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, (all_hidden_states, all_unnormed_states), all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=(all_hidden_states, all_unnormed_states),
            attentions=all_self_attns,
        )



# PAI code for visual attention excitation https://github.com/LALBJ/PAI/tree/master
# With this implementation the output_attention needs to be set to true to avoid problems
# Due to attention implementation LlamaAttn VS LlamaSdpaAttn...
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import types
def llama_attn_new_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print(self.alpha)
        if hasattr(self, "hook_manager"):
            attn_weights = self.hook_manager("before_attn_mask", ret = attn_weights)
        # exit()
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        if hasattr(self, "hook_manager"):
            attn_weights = self.hook_manager("after_attn_mask", ret = attn_weights)
        allow_modify = kwargs.get('allow_modify', False) or self.always_modify
        if not allow_modify: # original llamaattn implementation
            img_start_idx = self.img_start_idx
            img_end_idx = self.img_end_idx
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if hasattr(self, "hook_manager"):
                attn_weights = self.hook_manager("after_softmax", ret = attn_weights)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            if hasattr(self, "hook_manager"):
                attn_output = self.hook_manager("matmul_v", ret = attn_output)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)

            if hasattr(self, "hook_manager"):
                attn_output = self.hook_manager("attn_projected", ret = attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value
        ## PAI's modification
        if hasattr(self, "use_attn"):
            use_attn = self.use_attn
            img_start_idx = self.img_start_idx
            img_end_idx = self.img_end_idx
        else:
            use_attn = False

        if hasattr(self, "use_cfg"):
            use_cfg = self.use_cfg
        else:
            use_cfg = False

        if use_attn and not use_cfg:
            if self.excite_last_visual:
                attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
                    attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * self.alpha
                    + attn_weights[:, :, -1, img_start_idx:img_end_idx]
                )
            else:
                attn_weights[:, :, img_end_idx:, img_start_idx:img_end_idx] = (
                    attn_weights[:, :, img_end_idx:, img_start_idx:img_end_idx].abs() * self.alpha
                    + attn_weights[:, :, img_end_idx:, img_start_idx:img_end_idx]
                )
            
        ## PAI's modification
        
        # upcast attention to fp32

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if hasattr(self, "hook_manager"):
            attn_weights = self.hook_manager("after_softmax", ret = attn_weights)
        ### ACT's modification
        if self.log_bos is not None:
            self.log_bos = attn_weights[0, :, -1, 0].mean().item()
        if self.log_visual is not None:
            self.log_visual = attn_weights[0, :, -1, img_start_idx:img_end_idx].sum(dim=1).mean().item()
        if self.log_text is not None:
            self.log_text = attn_weights[0, :, -1, img_end_idx:].sum(dim=1).mean().item()
        if self.use_vact != 'none':
            threshold = self.threshold / attn_weights.shape[-1]
            
            
            if self.use_vact == 'text_sinks':
                new_average = None
            else:
                if attn_weights.shape[2] == attn_weights.shape[-1]:
                    first_in = True
                else:
                    first_in = False

                if first_in:
                    self.attn_cache = None
                    assert attn_weights.shape[0] == 1, 'For now only single batch operation is implementated'
                    new_average = None
                    self.embs_cache = F.normalize(hidden_states, p=2, dim=-1)  
                    self.sim_cache = None
                else:
                    avg_attn_scores = self.attn_cache
                    B, H, _, N = attn_weights.shape

                    new_attn_score = attn_weights.clone().squeeze(0).squeeze(1)

                    counts = torch.arange(N, 0, -1).to(attn_weights.device)  # This creates a tensor [N+1, N, ..., 1]

                    expanded_attn_scores = torch.nn.functional.pad(avg_attn_scores, (0, 1))  # Pad and reshape
                    pre_new_counts = counts[1:]  # [N, N-1, ..., 1]
                    pre_new_counts = pre_new_counts.expand(H, N - 1) 
                    expanded_pre_new_counts = torch.nn.functional.pad(pre_new_counts, (0, 1), value=1)
                    # Ensure counts tensor matches the required shape for broadcasting
                    counts_reshaped = counts.view(1, N)
                    # Compute the running sum, ensuring all dimensions match for broadcasting
                    running_sum = expanded_pre_new_counts * expanded_attn_scores + new_attn_score
                    new_average = running_sum / counts_reshaped
                    
                    # The following code keeps track of all hidden states in case of using cache when generate multiple tokens
                    normed_states = F.normalize(hidden_states, p=2, dim=-1)  
                    self.embs_cache = torch.cat((self.embs_cache, normed_states), dim=1)

            
            args_dict = {
                "attention_map": attn_weights,
                "threshold": threshold,
                "beta": self.beta,
                "include_bos": self.include_bos,
                "sink_criteria": self.sink_criteria,
                'average_cache': new_average,
            }
            
            if hasattr(self, "v_beta") and hasattr(self, "v_threshold"):
                v_threshold = self.v_threshold / attn_weights.shape[-1]
                
                modify_method = adjust_attention_map_vt
                
                text_token_mask = torch.ones(attn_weights.shape[-1], dtype=torch.bool)
                text_token_mask[img_start_idx:img_end_idx] = False
                args_dict = {
                    "attention_map": attn_weights,
                    "text_threshold": threshold,
                    "text_beta": self.beta,
                    "visual_threshold": v_threshold,
                    "visual_beta": self.v_beta,
                    "text_mask": text_token_mask,
                    "include_bos": self.include_bos,
                    "sink_criteria": self.sink_criteria,
                    'average_cache': new_average,
                    "include_bos_sink": self.include_bos_sink,
                    "log_sink": self.log_sink,
                    "modify_last_only": self.modify_last_only,
                }

            else:
                modify_method = adjust_attention_map

            if self.sink_criteria is not None:
                args_dict["sink_criteria"] = self.sink_criteria
                if self.sink_criteria == 'trim' or self.sink_criteria == 'vtrim':
                    args_dict["layer_embs"] = self.embs_cache
                    args_dict["sink_t"] = self.sink_t
                    args_dict['sim_cache'] = self.sim_cache

        if self.use_act:
            threshold = self.threshold / attn_weights.shape[-1]

            attn_weights = adjust_attention_map(attn_weights, threshold=threshold, beta=self.beta, include_bos = self.include_bos)
        elif self.use_vact == 'vt':
            sink_mask = torch.zeros(attn_weights.shape[-1], dtype=torch.bool)
            sink_mask[img_start_idx:img_end_idx] = True
            excite_mask = ~sink_mask
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask
        elif self.use_vact == 'tv':
            # tact
            sink_mask = torch.ones(attn_weights.shape[-1], dtype=torch.bool)
            sink_mask[img_start_idx:img_end_idx] = False
            excite_mask = ~sink_mask
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask
        elif self.use_vact == 'vv':
            sink_mask = torch.zeros(attn_weights.shape[-1], dtype=torch.bool)
            sink_mask[img_start_idx:img_end_idx] = True
            excite_mask = sink_mask
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask
        elif self.use_vact == 'tt':
            sink_mask = torch.ones(attn_weights.shape[-1], dtype=torch.bool)
            sink_mask[img_start_idx:img_end_idx] = False
            excite_mask = sink_mask

            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask
        elif self.use_vact == 'ta':
            sink_mask = torch.ones(attn_weights.shape[-1], dtype=torch.bool)
            sink_mask[img_start_idx:img_end_idx] = False
            excite_mask = None
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask
        elif self.use_vact == 'va':
            sink_mask = torch.zeros(attn_weights.shape[-1], dtype=torch.bool)
            sink_mask[img_start_idx:img_end_idx] = True
            excite_mask = None
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask

        elif self.use_vact == 'av':
            excite_mask = torch.zeros(attn_weights.shape[-1], dtype=torch.bool)
            excite_mask[img_start_idx:img_end_idx] = True
            sink_mask = None
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask

        elif self.use_vact == 'at':
            excite_mask = torch.ones(attn_weights.shape[-1], dtype=torch.bool)
            excite_mask[img_start_idx:img_end_idx] = False
            sink_mask = None
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask

        elif self.use_vact == 'aa':
            excite_mask = None
            sink_mask = None
            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask

        elif self.use_vact == 'bosv':
            sink_mask = torch.zeros(attn_weights.shape[-1], dtype=torch.bool)
            sink_mask[0] = True
            excite_mask = torch.zeros(attn_weights.shape[-1], dtype=torch.bool)
            excite_mask[img_start_idx:img_end_idx] = True

            args_dict['sink_mask'] = sink_mask
            args_dict['excite_mask'] = excite_mask

        elif self.use_vact == 'text_sinks':
            excite_mask = torch.ones(attn_weights.shape[-1], dtype=torch.bool)
            args_dict['excite_mask'] = excite_mask
            modify_method = adjust_reduce_bos
        elif self.use_vact != 'none':
            raise NotImplementedError

        else:
            pass
        if self.use_vact != 'none':
            out = modify_method(**args_dict)
            self.attn_cache = out['averages']
            self.sim_cache = out['similarity_matrix']
            if self.log_sink is not None:
                self.log_sink = out['sink_sum'] / 32

        ### ACT's modification


        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if hasattr(self, "hook_manager"):
            attn_output = self.hook_manager("matmul_v", ret = attn_output)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if hasattr(self, "hook_manager"):
            attn_output = self.hook_manager("attn_projected", ret = attn_output)

        if not output_attentions:
            attn_weights = None

        # print('max', attn_output[:, -1, :].max())


        return attn_output, attn_weights, past_key_value

def llama_modify(model, start_layer, end_layer, use_attn, alpha, use_cfg,
                 img_start_idx, img_end_idx, use_act=False, threshold=None, beta = None, use_vact='none', 
                 include_bos = False, v_beta=None, v_threshold = None, sink_criteria = None, sink_t = None, 
                 always_modify = False, excite_last_visual = True, include_bos_sink=False, log_bos=False, 
                 log_sink=False, log_visual=False, log_text=False, modify_last_only=False):
    
    print('modified attention')
    
    for i in range(start_layer, end_layer):
        model.model.layers[i].self_attn.use_attn = use_attn
        model.model.layers[i].self_attn.alpha = alpha
        model.model.layers[i].self_attn.use_cfg = use_cfg
        model.model.layers[i].self_attn.img_start_idx = img_start_idx
        model.model.layers[i].self_attn.img_end_idx = img_end_idx
        model.model.layers[i].self_attn.use_act = use_act
        model.model.layers[i].self_attn.use_vact = use_vact
        # model.model.layers[i].self_attn.use_tact = use_vact
        model.model.layers[i].self_attn.threshold = threshold
        model.model.layers[i].self_attn.beta = beta
        model.model.layers[i].self_attn.include_bos = include_bos
        model.model.layers[i].self_attn.sink_criteria = sink_criteria
        model.model.layers[i].self_attn.sink_t = sink_t
        model.model.layers[i].self_attn.always_modify = always_modify
        model.model.layers[i].self_attn.attn_cache = None
        model.model.layers[i].self_attn.embs_cache = None
        model.model.layers[i].self_attn.sim_cache = None
        model.model.layers[i].self_attn.include_bos_sink = include_bos_sink

        model.model.layers[i].self_attn.log_bos = 0 if log_bos else None
        model.model.layers[i].self_attn.log_sink = 0 if log_sink else None
        model.model.layers[i].self_attn.log_visual = 0 if log_visual else None
        model.model.layers[i].self_attn.log_text = 0 if log_text else None
        model.model.layers[i].self_attn.modify_last_only = modify_last_only


        # NOTE: In original PAI implementation, only the last token attention is being modified, 
        # which wouldnt work when using loss-based evaluation method (where potential answer tokens are given in inputs)
        # Here I modified it such that at loss-based eval, all token attention after visual end are modified
        # This would lead to slightly different model embedding as question tokens in shallow layers already have enhanced visual
        # information that would flow to latter layers. 
        model.model.layers[i].self_attn.excite_last_visual = excite_last_visual
        if v_beta is not None and v_threshold is not None:
            model.model.layers[i].self_attn.v_threshold = v_threshold
            model.model.layers[i].self_attn.v_beta = v_beta
            model.model.layers[i].self_attn.forward = types.MethodType(llama_attn_new_forward, model.model.layers[i].self_attn)
            
        else:
            model.model.layers[i].self_attn.forward = types.MethodType(llama_attn_new_forward, model.model.layers[i].self_attn)


def adjust_attention_map(attention_map, threshold, beta, include_bos=False, sink_mask=None, excite_mask=None):
    assert 'this method should not be used for text generation more than 1 token'
    device = attention_map.device
    num_heads, seq_len = attention_map.shape[1], attention_map.shape[2]

    # If masks are not provided, assume all positions can be sinks/excited
    if sink_mask is None:
        sink_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else: 
        sink_mask = sink_mask.to(dtype=torch.bool, device=device)
    if excite_mask is None:
        excite_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else:
        excite_mask = excite_mask.to(dtype=torch.bool, device=device)

    # Calculate the column sums and averages
    sums = torch.sum(attention_map[0], dim=1)  # Sum over rows for each column in each head
    desc_count = torch.arange(seq_len, 0, -1).float().to(device)  # Descending count from seq_len to 1
    averages = sums / desc_count.unsqueeze(0)  # Broadcast desc_count over num_heads dimension

    # Identify columns that exceed the threshold and are marked by sink_mask, ignoring the first token if needed
    too_much_attention = (averages > threshold) & sink_mask & (torch.arange(seq_len, device=device) != 0)
    # print(too_much_attention.sum())

    # Reduce attention directly in the attention map where attention is too much
    attention_mask = too_much_attention.unsqueeze(1).expand(-1, seq_len, -1)
    reduced_amount = attention_map[0] * (1 - beta) * attention_mask
    attention_map[0][attention_mask] *= beta
    total_reduced = reduced_amount.sum(dim=2, keepdim=True)

    # Calculate proportional distribution factors
    # Avoid including reduced columns in the distribution factors and apply excite_mask
    valid_attention = attention_map[0] * (~attention_mask)
    if not include_bos:
        valid_attention *= (torch.arange(seq_len, device=device) != 0).unsqueeze(0).unsqueeze(0)
    # Currently, excite mask servers as furtehr restriction. It cannot allow sink tokens to get redistribution
    valid_attention *= excite_mask.unsqueeze(0).unsqueeze(0)

    row_sums = valid_attention.sum(dim=2, keepdim=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    distribution_factors = valid_attention / row_sums

    # Distribute the reduced attention proportionally
    redistributed_attention = total_reduced * distribution_factors
    attention_map[0] += redistributed_attention

    return attention_map

def adjust_attention_map_vt(attention_map, text_threshold, text_beta, visual_threshold,  visual_beta, text_mask, 
                            include_bos=False, sink_mask=None, excite_mask=None, max_sinks=None, sink_criteria=None, 
                            layer_embs=None, sink_t = None, average_cache = None, sim_cache = None, include_bos_sink = False,
                            log_sink=None, modify_last_only=False):
    return_dict = {}
    device = attention_map.device
    num_heads, seq_len = attention_map.shape[1], attention_map.shape[3]
    # print(text_threshold, text_beta, visual_threshold,  visual_beta,)
    if sink_mask is None:
        sink_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else: 
        sink_mask = sink_mask.to(dtype=torch.bool, device=device)
        
    if excite_mask is None:
        excite_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else:
        excite_mask = excite_mask.to(dtype=torch.bool, device=device)

    text_mask = text_mask.to(dtype=torch.bool, device=device)
    visual_mask = ~text_mask
    
    # Calculate column sums and averages separately for text and visual tokens
    # Here the attn cache which would always be a square matrix should be used
    if average_cache is None:
        sums = torch.sum(attention_map[0], dim=1)  # Sum over rows for each column in each head
        desc_count = torch.arange(seq_len, 0, -1).float().to(device)  # Descending count from seq_len to 1
        averages = sums / desc_count.unsqueeze(0)  # Broadcast desc_count over num_heads dimension
    else:
        averages = average_cache

    # Initial conditions based on text and visual thresholds, and respective masks
    if include_bos_sink:
        text_condition = (averages > text_threshold) & sink_mask & text_mask
    else:
        text_condition = (averages > text_threshold) & sink_mask & text_mask & (torch.arange(seq_len, device=device) != 0)
    visual_condition = (averages > visual_threshold) & sink_mask & visual_mask & (torch.arange(seq_len, device=device) != 0)
    
    if sink_criteria is not None:
        norms = layer_embs
        # TODO: Optimization here, maybe we dont have to compute the whole matrix, but just the candidate ones
        if sim_cache is None:
            similarity_matrix = torch.bmm(norms, norms.transpose(1, 2))
        else:
            new_state_norm = norms[:, -1, :]
            old_states = norms[:, :-1, :]
            new_similarities = torch.matmul(new_state_norm, old_states.transpose(-2, -1))
            updated_matrix = torch.cat([sim_cache, new_similarities.transpose(-2, -1)], dim=2)
            new_row = torch.cat([new_similarities, torch.ones(1, 1, 1, device=norms.device)], dim=2)  # self-similarity is 1
            similarity_matrix = torch.cat([updated_matrix, new_row], dim=1)
        
        if sink_criteria == 'trim':
            text_condition = filter_by_similarity(text_condition, similarity_matrix, sim_t=sink_t)
            visual_condition = filter_by_similarity(visual_condition, similarity_matrix, sim_t=sink_t)
        elif sink_criteria == 'vtrim':
            visual_condition = filter_by_similarity(visual_condition, similarity_matrix, sim_t=sink_t)
            

    # Combine conditions
    initial_condition = text_condition | visual_condition
    
    # Applying the max_sinks constraint
    if max_sinks is not None:
        _, top_indices = torch.topk(averages * initial_condition.float(), max_sinks, dim=1)
        too_much_attention = torch.zeros_like(averages, dtype=torch.bool)
        too_much_attention.scatter_(1, top_indices, 1)
    else:
        too_much_attention = initial_condition

   
    if average_cache is None:
        attention_mask = too_much_attention.unsqueeze(1).expand(-1, seq_len, -1)
    else:
        attention_mask = too_much_attention.unsqueeze(1).expand(-1, 1, -1)
    
    if modify_last_only:
        attention_mask = attention_mask.clone() # get rid of the expand behaviour where set one row expand to all values
        
        attention_mask[:, :-1, :] = False


    expanded_text_mask = text_mask.unsqueeze(0).expand(num_heads, -1, -1)
    
    reduced_amount_text = attention_map[0] * attention_mask
    reduced_amount_text = reduced_amount_text * expanded_text_mask 
    if log_sink is not None:
        return_dict['sink_sum'] = reduced_amount_text[:, -1, :].sum().item()
        

    reduced_amount_text = reduced_amount_text * (1 - text_beta)
    

    expanded_visual_mask = visual_mask.unsqueeze(0).expand(num_heads, -1, -1)

    reduced_amount_visual = attention_map[0] * attention_mask
    reduced_amount_visual = reduced_amount_visual * expanded_visual_mask 
    if log_sink is not None:
        return_dict['sink_sum'] += reduced_amount_visual[:, -1, :].sum().item()
    reduced_amount_visual = reduced_amount_visual * (1 - visual_beta)

    attention_map[0][attention_mask & expanded_text_mask] *= text_beta
    attention_map[0][attention_mask & expanded_visual_mask] *= visual_beta
    
    # Sum the reduced amounts
    total_reduced = reduced_amount_text.sum(dim=2, keepdim=True) + reduced_amount_visual.sum(dim=2, keepdim=True)

    # Calculate proportional distribution factors
    valid_attention = attention_map[0] * (~attention_mask) * excite_mask.unsqueeze(0).unsqueeze(0)
    row_sums = valid_attention.sum(dim=2, keepdim=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    distribution_factors = valid_attention / row_sums

    # Distribute the reduced attention proportionally
    redistributed_attention = total_reduced * distribution_factors
    attention_map[0] += redistributed_attention

    return_dict.update({'attention_map': attention_map,
                   'averages': averages,
                   'similarity_matrix': None
                   })
    
    if sink_criteria is not None:
        return_dict['similarity_matrix'] = similarity_matrix
    return return_dict

def filter_by_similarity(condition, similarity_matrix, sim_t=0.05, trim_ratio=0.01):
    """
    Filters tokens based on the trimmed mean of their similarity distribution exceeding sim_t.
    Args:
        condition (torch.Tensor): Boolean tensor indicating potential sink tokens.
        similarity_matrix (torch.Tensor): Tensor of shape [batch, seq_len, seq_len] containing similarity scores.
        sim_t (float): Threshold for the trimmed mean of similarities to consider a token as a sink.
        trim_ratio (float): Proportion of values to trim from each end of the similarity distribution.

    Returns:
        torch.Tensor: Updated condition tensor.
    """
    # Sort similarity values along the last dimension
    sorted_similarities, _ = similarity_matrix.sort(dim=-1)
    # Determine number of elements to trim from each end
    num_elements = sorted_similarities.size(-1)
    trim_count = int(num_elements * trim_ratio)

    # Apply trimming
    trimmed_similarities = sorted_similarities[:, :, trim_count:num_elements-trim_count]
    
    # Compute the mean across the trimmed distributions
    trimmed_mean_similarities = trimmed_similarities.mean(dim=-1)
    # Update the condition based on the threshold
    updated_condition = (trimmed_mean_similarities < sim_t) & condition

    return updated_condition

def adjust_reduce_bos(attention_map, text_threshold, text_beta, visual_threshold,  visual_beta, text_mask, 
                            include_bos=False, sink_mask=None, excite_mask=None, max_sinks=None, sink_criteria=None, 
                            layer_embs=None, sink_t = None, average_cache = None, sim_cache = None):
    
    device = attention_map.device
    num_heads, seq_len = attention_map.shape[1], attention_map.shape[3]
    # print(text_threshold, text_beta, visual_threshold,  visual_beta,)

    if sink_criteria == 'tov':
        excite_mask = ~text_mask
    
    if sink_mask is None:
        sink_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else: 
        sink_mask = sink_mask.to(dtype=torch.bool, device=device)
        
    if excite_mask is None:
        excite_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    else:
        excite_mask = excite_mask.to(dtype=torch.bool, device=device)


    text_mask = text_mask.to(dtype=torch.bool, device=device)
    visual_mask = ~text_mask
    # Initial conditions based on text and visual thresholds, and respective masks
    text_condition = torch.zeros((num_heads, seq_len), dtype=torch.bool, device=device)
    text_condition[:, 0] = True
    if sink_criteria == 'two_text':
        text_condition[:, 12] = True

    visual_condition = torch.zeros((num_heads, seq_len), dtype=torch.bool, device=device)
    
    # Combine conditions
    initial_condition = text_condition | visual_condition
    # Applying the max_sinks constraint

    too_much_attention = initial_condition

    # Reduce attention where it's too much, using separate betas
    # attention_mask defines sink tokens found according to above conditions
    attention_mask = too_much_attention.unsqueeze(1).expand(-1, attention_map.shape[2], -1)

    expanded_text_mask = text_mask.unsqueeze(0).expand(num_heads, -1, -1)
    # expanded_text_mask = text_mask.unsqueeze(1).expand(num_heads, -1, -1)
    reduced_amount_text = attention_map[0] * (1 - text_beta)

    reduced_amount_text = reduced_amount_text * attention_mask

    # Apply the text mask
    reduced_amount_text = reduced_amount_text * expanded_text_mask 

    expanded_visual_mask = visual_mask.unsqueeze(0).expand(num_heads, -1, -1)

    reduced_amount_visual = attention_map[0] * (1 - visual_beta)
    reduced_amount_visual = reduced_amount_visual * attention_mask

    reduced_amount_visual = reduced_amount_visual * expanded_visual_mask 
    attention_map[0][attention_mask & expanded_text_mask] *= text_beta
    attention_map[0][attention_mask & expanded_visual_mask] *= visual_beta

    # Sum the reduced amounts
    total_reduced = reduced_amount_text.sum(dim=2, keepdim=True) + reduced_amount_visual.sum(dim=2, keepdim=True)

    # Calculate proportional distribution factors
    valid_attention = attention_map[0] * (~attention_mask) * excite_mask.unsqueeze(0).unsqueeze(0)
    row_sums = valid_attention.sum(dim=2, keepdim=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    distribution_factors = valid_attention / row_sums

    # Distribute the reduced attention proportionally
    redistributed_attention = total_reduced * distribution_factors
    if sink_criteria != 'reduce':
        attention_map[0] += redistributed_attention

    if sink_criteria is None:
        return attention_map, None, None
    return attention_map, None, None
