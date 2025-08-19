import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import numpy as np

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, ModelOutput
from transformers.generation import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from experiments.llava.constants import IMAGE_TOKEN_INDEX
from experiments.utils.helper_functions import *
from experiments.utils.attention_resampler import aggregate_layer_attention, attention_sampling, get_highlight_mask, prepare_attention_mask
from experiments.API.API_LLaVA.main import revise_mask, toImg, invtrans
from experiments.llava.mm_utils import process_images
from PIL import Image


def modify_model(model, args):
    from experiments.llava.model.language_model.llava_llama import llama_modify
    if getattr(args, 'heatmap', None) == 'api_llava':
        llama_modify(model, args.api_llava_layer, 32, False, 0, False, img_start_idx=35, img_end_idx=35 + 576)  


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    ## For contrastive decoding initial
    return_cd_scores = model_kwargs.get('return_cd_scores', None)
    output_logits = model_kwargs.get('output_logits', False) # logits is different from scores which will be processed according to sampling strategy such as top_k
    logits_tuple = () if (return_dict_in_generate and output_logits) else None
    
    override_mask = model_kwargs.get('override_mask', None)
    resample_args = model_kwargs.get('resample_args', None)

    print_logits = False
    scores_cd = {} if (return_dict_in_generate and output_scores and return_cd_scores) else None
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

            
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if override_mask is not None:
            model_inputs['override_mask'] = override_mask
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            # output_attentions=True,
            output_hidden_states=output_hidden_states,
        )

        
        ### The code below is hihgly similar to that in model_generate, it can be used as alternatives when open-end generation is needed
        if resample_args is not None:
            
            attentions = [outputs.attentions]
            if resample_args.normalize_mask == 'norm_enhance':
                resample_args.map_constructor = 'original'
            pad_mask = resample_args.pad_mask
            if resample_args.heatmap == None or resample_args.heatmap == 'attention':
                shallow_aggregate, deep_aggregate = aggregate_layer_attention(attentions, target_size=resample_args.image_pad_pil.size, attention_mode='max', vis_start=resample_args.vis_start, deep_layer=resample_args.deep_layer, map_constructor=resample_args.map_constructor, single_layer=resample_args.single_layer)
                if resample_args.normalize_mask != 'norm_enhance':
                    heatmap = deep_aggregate * pad_mask
                height, width = pad_mask.shape[:2]
                target_size = (height, width)

                if resample_args.normalize_mask == 'norm_enhance':
                    heatmap = deep_aggregate
                    heatmap_th = revise_mask(torch.Tensor(heatmap), kernel_size=resample_args.kernel_size, enhance_coe=resample_args.enhance_coe)
                    heatmap = heatmap_th.detach().cpu()
                    mask = toImg(heatmap.reshape(1,24,24))
                    interpolate_method = getattr(Image, "BICUBIC") 
                    mask = invtrans(mask, resample_args.image_pad_pil, method = interpolate_method)
                    heatmap = mask

            elif resample_args.heatmap == 'iterative':
                attentions = [outputs.attentions]
                shallow_ori_aggregate, deep_ori_aggregate = aggregate_layer_attention(attentions, target_size=resample_args.image_pad_pil.size, attention_mode='max', vis_start=resample_args.vis_start, deep_layer=resample_args.deep_layer, 
                                                        map_constructor='original', single_layer=resample_args.single_layer)
                iterative_attn_mean_sum = np.zeros((deep_ori_aggregate.shape))
                added_cnt = np.zeros((deep_ori_aggregate.shape))
                iterative_mask = np.ones((deep_ori_aggregate.shape)).astype(bool)
                it = 0
                while True:
                    if resample_args.iterative_search is not None:
                        if resample_args.iterative_search == 'token_mask':
                            deep_mean = deep_ori_aggregate / (len(attentions[0]) - resample_args.deep_layer)

                            # Get highlighted attention regions in the current map
                            added_cnt += iterative_mask
                            iterative_attn_mean_sum += deep_mean
                            iterative_attn_avg = iterative_attn_mean_sum / added_cnt

                            highlight_mask = get_highlight_mask(deep_mean)

                            iterative_mask = np.bitwise_and(iterative_mask, ~highlight_mask)

                            attention_mask_new = prepare_attention_mask(len(input_ids[0])+575, ~iterative_mask, resample_args.vis_start)
                            attention_mask_new = torch.Tensor(attention_mask_new).unsqueeze(dim=0).to('cuda')

                            model_kwargs_resample = model_kwargs.copy()

                            model_inputs_resample = self.prepare_inputs_for_generation(input_ids, **model_kwargs_resample)
                            model_inputs_resample['override_mask'] = attention_mask_new

                            resample_outputs = self(
                                **model_inputs_resample,
                                return_dict=True,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                allow_modify = False,
                            )

                            attentions = [resample_outputs.attentions]

                            shallow_ori_aggregate, deep_ori_aggregate = aggregate_layer_attention(attentions, target_size=resample_args.image_pad_pil.size, attention_mode='max', vis_start=resample_args.vis_start, deep_layer=resample_args.deep_layer, 
                                                        map_constructor='original', single_layer=resample_args.single_layer)
                            it += 1

                            exit_flag = False
                            if resample_args.exit_criteria == 'sum_attention':
                                deep_sum = deep_ori_aggregate.sum().item() / (len(attentions[0]) - resample_args.deep_layer)
                                if deep_sum <= resample_args.exit_threshold:
                                    exit_flag = True
                            else:
                                raise NotImplementedError
                            if it >= resample_args.max_iteration:
                                exit_flag = True
                            if exit_flag:
                                height, width = pad_mask.shape[:2]
                                target_size = (height, width)
                                
                                if resample_args.normalize_mask == 'norm_enhance':
                                    heatmap_th = revise_mask(torch.Tensor(iterative_attn_avg), kernel_size=resample_args.kernel_size, enhance_coe=resample_args.enhance_coe)
                                    heatmap = heatmap_th.detach().cpu()
                                    mask = toImg(heatmap.reshape(1,24,24))
                                    interpolate_method = getattr(Image, "BICUBIC")
                                    mask = invtrans(mask, resample_args.image_pad_pil, method = interpolate_method)
                                    heatmap = mask
                                else:
                                    raise NotImplementedError
                                
                                break

            if resample_args.resample_method == 'magnify':
                if isinstance(heatmap, torch.Tensor):
                    heatmap = heatmap.numpy()
                elif isinstance(heatmap, Image.Image):
                    heatmap = np.array(heatmap)
                height, width = pad_mask.shape[:2]
                target_size = (height, width)
                resampled_image = attention_sampling(np.array(resample_args.image_pad_pil), heatmap, target_size, method = 'bilinear')
                resampled_image = resampled_image.astype(np.uint8)
                resampled_image = Image.fromarray(resampled_image, 'RGB')
                pass
            else:
                raise NotImplementedError

            resampled_image = process_images([resampled_image], resample_args.image_processor, self.config)
            

            model_kwargs_resample = model_kwargs.copy()
            model_kwargs_resample['use_cache'] = False
            model_kwargs_resample['past_key_values'] = None
            model_kwargs_resample['images'] = resampled_image.to(dtype=model_kwargs_resample['images'].dtype, device='cuda', non_blocking=True)
            model_inputs_resample = self.prepare_inputs_for_generation(input_ids, **model_kwargs_resample)
            outputs_resample = self(
                **model_inputs_resample,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                allow_modify = False,
            )
            
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        if resample_args is not None:
            next_token_logits = outputs_resample.logits[:, -1, :].detach().clone()
            outputs_resample = outputs # frees the memory 

        if print_logits:
            print('base', next_token_logits[0, 3869].item(), next_token_logits[0, 1939].item())

        
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)


        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
       
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            if return_cd_scores or output_logits:
                return PMDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    scores_cd = scores_cd,
                    logits = logits_tuple,
                    # scores_cd = scores
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )

    else:
        return input_ids
    
@dataclass
class PMDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    scores_cd: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None

def _update_model_kwargs_for_cd_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

def evolve_pm_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaSdpaAttention
    # A slightly hacky method to add **kwargs to sdpa attn forward call allowing for dynamic control on the run
    LlamaSdpaAttention.original_forward = LlamaSdpaAttention.forward
    def forward_with_kwargs(self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
        ):
        return self.original_forward(hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache)  
    LlamaSdpaAttention.forward = forward_with_kwargs

    transformers.generation.utils.GenerationMixin._update_model_kwargs_for_cd_generation = _update_model_kwargs_for_cd_generation
