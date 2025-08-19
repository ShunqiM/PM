import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import argparse
import datetime
import json
from pathlib import Path

from llava.hook import HookManager

def init_hookmanager(module):
    module.hook_manager = HookManager()

class MaskHookLogger(object):
    def __init__(self, model, device, multiple_layer=None):
        self.current_layer = 0
        self.device = device
        self.attns = []
        self.projected_attns = []
        self.image_embed_range = []
        self.index = []
        self.model = model
        self.multiple_layer = multiple_layer
        
    @torch.no_grad()
    def compute_attentions(self, ret):
        assert len(self.image_embed_range) > 0
        st, ed = self.image_embed_range[-1]
        image_attention = ret[:,:,-1,st:ed].detach()
        image_attention = image_attention.mean(dim = 1)
        self.attns.append(image_attention) # [b, k]
        return ret
        
    @torch.no_grad()
    def compute_projected_attentions(self, ret):
        assert len(self.image_embed_range) > 0
        st, ed = self.image_embed_range[-1]
        image_attention = ret[:,-1,st:ed].detach() # [b, k, d]
        self.projected_attns.append(image_attention) # [b, k, d]
        return ret
        
    @torch.no_grad()
    def compute_attentions_withsoftmax(self, ret):
        # print(ret.shape, self.image_embed_range)
        assert len(self.image_embed_range) > 0
        st, ed = self.image_embed_range[-1]
        image_attention = ret[:,:,-1,st:ed].detach()
        image_attention = image_attention.softmax(dim = -1)
        image_attention = image_attention.mean(dim = 1)
        # print('image_attention place', image_attention.shape)
        # exit()

        self.attns.append(image_attention) # [b, k]
        return ret
    
    @torch.no_grad()
    def compute_logits_index(self, ret):
        next_token_logits = ret[:, -1, :]
        index = next_token_logits.argmax(dim=-1)
        self.index.append(index.item())
        return ret
    
    @torch.no_grad()
    def finalize(self): # This will be used for attention map compuatation
        attns = torch.cat(self.attns, dim = 0).to(self.device) 
        # print('attns shape',attns.shape)
        if self.multiple_layer is not None:
            n_img_token = attns.shape[-1]
            # print(n_img_token, self.multiple_layer)
            attns = attns.reshape(-1, self.multiple_layer, n_img_token)
            attns = attns.mean(dim=1)
            # print('attns new shape',attns.shape)

        # exit()
        return attns
    
    @torch.no_grad()
    def finalize_projected_attn(self, norm_weight, proj):
        assert len(self.index) == len(self.projected_attns)
        mask = []
        for i in range(-4,-2):
            index = self.index[i]
            attns = self.projected_attns[i].to(self.device) # 1,k,d
            input_dtype = attns.dtype
            attns_var = attns.to(torch.float32).sum(dim = 1).pow(2).mean(-1, keepdim=True)# 1,d
            attns_var = attns_var.unsqueeze(1)# 1,1,d
            normalized_attns = attns * torch.rsqrt(attns_var + 1e-6) # 1,k,d
            normalized_attns = norm_weight.to(normalized_attns.device) * normalized_attns.to(input_dtype) # 1,k,d
            logits = proj(normalized_attns) 
            max_logits = logits[0,:,index] # k
            mask.append(max_logits)

        mask = torch.stack(mask, dim = 0)

        return mask.mean(dim = 0)
        
    def reinit(self):
        self.attns = []
        self.projected_attns = []
        self.image_embed_range = []
        self.index = []
        torch.cuda.empty_cache()

    def log_image_embeds_range(self, ret):
        self.image_embed_range.append(ret[0][0])
        return ret

def hook_logger(model, device, layer_index = 20, single_layer=True):
    """Hooks a projected residual stream logger to the model."""
    # print(len(model.model.layers))
    if single_layer:
        init_hookmanager(model.model.layers[layer_index].self_attn)

        prs = MaskHookLogger(model, device)
        model.model.layers[layer_index].self_attn.hook_manager.register('after_attn_mask',
                                    prs.compute_attentions_withsoftmax)
    else:
        # init_hookmanager(model.model.layers[layer_index].self_attn)
        

        prs = MaskHookLogger(model, device, multiple_layer=len(model.model.layers) - layer_index)
        for i in range(layer_index, len(model.model.layers)):
            init_hookmanager(model.model.layers[i].self_attn)
            model.model.layers[i].self_attn.hook_manager.register('after_attn_mask',
                                        prs.compute_attentions_withsoftmax)

        # Register hooks for multiple layers
        # for i in range(layer_index, len(model.model.layers)):
        #     def hook_function_factory(layer_id):
        #         def hook_function(**kwargs):
        #             return prs.compute_attentions_withsoftmax(layer_id=layer_id, **kwargs)
        #         return hook_function

        #     hook_function = hook_function_factory(i)
        #     model.model.layers[i].self_attn.hook_manager.register('after_attn_mask', hook_function)

    # else:
    #     raise NotImplementedError

    model.hooklogger = prs
    
    return prs