import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments'))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from transformers import set_seed, AutoProcessor
from utils.metrics import calibrate_label_dict, get_prob_from_logits
from experiments.utils.pm_sample import evolve_pm_sampling, modify_model
evolve_pm_sampling()
import numpy as np
from PIL import Image
import math
import json
from experiments.utils.helper_functions import *
from experiments.utils.attention_resampler import *
from experiments.API.API_LLaVA.main import *
from experiments.utils.api_utils import init_clip
from experiments.API.API_CLIP.main import gen_mask as clip_gen_mask, blend_mask as clip_blend_mask, merge_mask as clip_merge_mask
from experiments.API.API_LLaVA.hook import hook_logger as llava_hook_logger
from experiments.API.API_LLaVA.functions import get_token_mapping, from_preanswer_to_mask as llava_from_preanswer_to_mask

from torchvision.transforms.functional import to_pil_image
from utils.utils_relevancy import construct_relevancy_map, SEPARATORS_LIST

Image.MAX_IMAGE_PIXELS = 933120000


from starter.env_getter import get_env
CKPT_DIR = get_env('CKPT')
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, return_padding=False):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.return_padding = return_padding
        if 'POPE' in self.image_folder:
            self.postfix = " Please answer this question with one word."
        else:
            self.postfix = ""

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + self.postfix)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        if self.return_padding:
            image_tensor, image_pil, pad_mask = process_images_return_pad([image], self.image_processor, self.model_config)
            image_tensor = image_tensor[0]
            pad_mask = np.array(pad_mask, dtype=int)
            return input_ids, image_tensor, image.size, image_pil, pad_mask

        else:
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)
    

def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def collate_fn_general(batch):
    # Unpack the items from each sample in the batch
    # zip(*batch) will give tuples of each element type in the dataset return
    items = list(zip(*batch))
    
    collated_items = []
    for item in items:
        if isinstance(item[0], torch.Tensor):
            collated_items.append(torch.stack(item, dim=0))
        else:
            collated_items.append(item)
    
    return tuple(collated_items)

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=8, 
                       return_pad_image = False,):
    assert batch_size == 1, "batch_size must be 1"

    collate_fn_arg = collate_fn_general if return_pad_image else collate_fn
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, return_padding=return_pad_image)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_arg)
    return data_loader

def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    if 'MME' in args.question_file:
        list_subsets = ["existence", "count", "position", "color", "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
        filtered_questions = [q for q in questions if q["category"] in list_subsets]
        questions = filtered_questions
    else: 
        pass

    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        print('Output file already exist')
        gen_files = [json.loads(q) for q in open(os.path.expanduser(answers_file), "r")]
        if len(gen_files) == len(questions):
            print('Command Return')
            exit()
        else:
            print('Overwrite incompete answer file')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    base_name, extension = os.path.splitext(answers_file)
    args_name = base_name + '-args'
    args_file = args_name + extension
    args_dict = vars(args)
    with open(args_file, 'w') as file:
        json.dump(args_dict, file, indent=4)

    compute_dtype = torch.float16
    if 'RLHF' in args.model_path:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name
        )

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=compute_dtype)
        image_processor = vision_tower.image_processor
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, subclass=True, load_8bit=args.load_8bit)
    modify_model(model, args)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # exit()
    if args.heatmap == 'api_clip':
        clip_model_dict = init_clip()
    elif args.heatmap == 'api_llava': # NOTE: currently we directly handle attention function modification in modify_model() method, in a non-compatible manner with pai modifications... Do we ever need to consider their compatibility? 
        llava_hl = llava_hook_logger(model, 'cuda', layer_index = args.api_llava_layer, single_layer=not args.aggreagte_api_attns)

    elif args.heatmap == 'relevancy_map':
        from llava.model import LcdLlavaLlamaForCausalLM ## override the model class to enable grad for sample function
        func_to_enable_grad = 'sample'
        setattr(LcdLlavaLlamaForCausalLM, func_to_enable_grad, torch.enable_grad(getattr(LcdLlavaLlamaForCausalLM, func_to_enable_grad)))

        model.enc_attn_weights = []
        model.enc_attn_weights_vit = []

        #outputs: attn_output, attn_weights, past_key_value
        def forward_hook(module, inputs, output): 
            if output[1] is None:
                print('output attention not logged')
                return output
            
            output[1].requires_grad_(True)
            output[1].retain_grad()
            model.enc_attn_weights.append(output[1])
            return output
        
        hooks_pre_encoder, hooks_encoder = [], []
        for layer in model.model.layers:
            hook_encoder_layer = layer.self_attn.register_forward_hook(forward_hook)
            hooks_pre_encoder.append(hook_encoder_layer)


    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, return_pad_image=args.image_resample)

    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    if args.save_resampled_image:
        image_save_folder = os.path.dirname(os.path.expanduser(args.answers_file))
        image_save_folder = os.path.join(image_save_folder, 'resampled_images')
        os.makedirs(image_save_folder, exist_ok=True)

    ith_batch = 0
    for inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        if args.image_resample:

            input_ids, image_tensor, image_sizes, image_pad_pil, pad_mask = inputs
            image_pad_pil = image_pad_pil[0]
            pad_mask = pad_mask[0]
        else:
            input_ids, image_tensor, image_sizes = inputs

        idx = line["question_id"]
        image_file = line["image"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        with torch.inference_mode(True):
            
            vis_start = torch.where(input_ids == -200)[1].item() # [1] get rid of row idx
            vis_end = vis_start + 24 * 24

            args_dict = {
                "input_ids": input_ids,
                "images": image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                "do_sample": True if args.temperature > 0 else False,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "use_cache": True,
                "output_scores": True,
                "return_dict_in_generate": True,
                "output_attentions": True,
            }
            
            
            if args.resample_during_sampling: 
                ## This moves everything relevant to resampling to the sample functions, enalbing the PM sampling to be performed during the generation process.
                ## Otherwise, setting resampl_during_sampling to False will still works for qa tasks, where only the first token will use the resampled technique.
                resample_args = argparse.Namespace()
                resample_args.image_pad_pil = image_pad_pil
                resample_args.resample_method = args.resample_method
                resample_args.normalize_mask = args.normalize_mask
                resample_args.image_processor = image_processor
                resample_args.vis_start=vis_start
                resample_args.deep_layer=args.deep_layer 
                resample_args.map_constructor=args.map_constructor
                resample_args.single_layer=args.single_layer
                resample_args.kernel_size=args.kernel_size
                resample_args.enhance_coe=args.enhance_coe
                resample_args.heatmap = args.heatmap
                resample_args.exit_criteria = args.exit_criteria
                resample_args.exit_threshold = args.exit_threshold
                resample_args.max_iteration = args.max_iteration
                resample_args.pad_mask = pad_mask
                resample_args.iterative_search = args.iterative_search
                args_dict['resample_args'] = resample_args

            if args.image_resample and args.heatmap == 'relevancy_map':
                with torch.inference_mode(False):
                    model_outputs = model.generate(**args_dict)
            else:
                model_outputs = model.generate(**args_dict)
            output_ids = model_outputs['sequences'].detach()
            scores = model_outputs['scores'][0].detach()
            attention = model_outputs['attentions']
            original_p_y = None
            if args.return_logits:
                tokens_naive, logits_raw = calibrate_label_dict(scores, tokenizer, return_logits=True)
                average_logis = scores[0].mean().item()
                original_scores = scores
        
            else:
                tokens_naive = calibrate_label_dict(scores, tokenizer)
            original_p_y = get_prob_from_logits(tokens_naive)

            if args.image_resample and not args.resample_during_sampling:
                ### resample_during_sampling is used for open-end generation, where we do not resample the image during the generation process.
                if args.normalize_mask == 'norm_enhance':
                    args.map_constructor = 'original'
                if args.heatmap == None or args.heatmap == 'attention':
                    shallow_aggregate, deep_aggregate = aggregate_layer_attention(attention, target_size=image_pad_pil.size, attention_mode='max', vis_start=vis_start, deep_layer=args.deep_layer, map_constructor=args.map_constructor, single_layer=args.single_layer)
                    if args.normalize_mask != 'norm_enhance':
                        heatmap = deep_aggregate * pad_mask
                    height, width = pad_mask.shape[:2]
                    target_size = (height, width)

                    if args.normalize_mask == 'norm_enhance':
                        heatmap = deep_aggregate
                        heatmap_th = revise_mask(torch.Tensor(heatmap), kernel_size=args.kernel_size, enhance_coe=args.enhance_coe)
                        heatmap = heatmap_th.detach().cpu()
                        mask = toImg(heatmap.reshape(1,24,24))
                        interpolate_method = getattr(Image, "BICUBIC") 
                        mask = invtrans(mask, image_pad_pil, method = interpolate_method)
                        heatmap = mask
                        
                    elif args.normalize_mask == 'regularization':
                        heatmap += args.resample_regularization
                    
                elif args.heatmap == 'iterative':
                    attentions = attention
                    shallow_ori_aggregate, deep_ori_aggregate = aggregate_layer_attention(attention, target_size=image_pad_pil.size, attention_mode='max', vis_start=vis_start, deep_layer=args.deep_layer, 
                                                           map_constructor='original', single_layer=args.single_layer)
                    iterative_attn_mean_sum = np.zeros((deep_ori_aggregate.shape))
                    added_cnt = np.zeros((deep_ori_aggregate.shape))
                    iterative_mask = np.ones((deep_ori_aggregate.shape)).astype(bool)
                    it = 0
                    while True:
                        if args.iterative_search is not None:
                            if args.iterative_search == 'token_mask':
                                deep_mean = deep_ori_aggregate / (len(attentions[0]) - args.deep_layer)
                                added_cnt += iterative_mask
                                iterative_attn_mean_sum += deep_mean
                                iterative_attn_avg = iterative_attn_mean_sum / added_cnt

                                highlight_mask = get_highlight_mask(deep_mean)

                                iterative_mask = np.bitwise_and(iterative_mask, ~highlight_mask)

                                attention_mask_new = prepare_attention_mask(len(input_ids[0])+575, ~iterative_mask, vis_start)
                                attention_mask_new = torch.Tensor(attention_mask_new).unsqueeze(dim=0).to('cuda')


                                iterative_arg_dict = args_dict.copy()
                                iterative_arg_dict['override_mask'] = attention_mask_new

                                model_outputs = model.generate(**iterative_arg_dict)

                                attentions = model_outputs['attentions']
                                scores = model_outputs['scores'][0]
                                tokens_naive = calibrate_label_dict(scores, tokenizer)
                                new_p_y = get_prob_from_logits(tokens_naive)

                                shallow_ori_aggregate, deep_ori_aggregate = aggregate_layer_attention(attentions, target_size=image_pad_pil.size, attention_mode='max', vis_start=vis_start, deep_layer=args.deep_layer, 
                                                           map_constructor='original', single_layer=args.single_layer)
                                it += 1

                                exit_flag = False
                                if args.exit_criteria == 'probability':
                                    if new_p_y[0] <= args.exit_threshold:
                                        exit_flag = True
                                elif args.exit_criteria == 'sum_attention':
                                    deep_sum = deep_ori_aggregate.sum().item() / (len(attentions[0]) - args.deep_layer)
                                    if deep_sum <= args.exit_threshold:
                                        exit_flag = True
                                else:
                                    raise NotImplementedError
                                if it >= args.max_iteration:
                                    exit_flag = True
                                if exit_flag:
                                    height, width = pad_mask.shape[:2]
                                    target_size = (height, width)
                                    

                                    if args.iterative_regularization:
                                        resample_regularization = args.resample_regularization / it
                                    else:
                                        resample_regularization = args.resample_regularization 

                                    if args.normalize_mask == 'norm_enhance':
                                        heatmap_th = revise_mask(torch.Tensor(iterative_attn_avg), kernel_size=args.kernel_size, enhance_coe=args.enhance_coe)
                                        heatmap = heatmap_th.detach().cpu()
                                        mask = toImg(heatmap.reshape(1,24,24))
                                        interpolate_method = getattr(Image, "BICUBIC")
                                        mask = invtrans(mask, image_pad_pil, method = interpolate_method)
                                        heatmap = mask

                                    elif args.normalize_mask == 'regularization':
                                        heatmap = resize_attention(iterative_attn_avg, target_size)
                                        heatmap = pad_mask * heatmap * (len(attentions[0]) - args.deep_layer)
                                        heatmap += resample_regularization

                                    else:
                                        raise NotImplementedError
                                    
                                    break
                elif args.heatmap == 'api_clip':
                    with torch.no_grad():
                        cls_mask, patch_mask = clip_gen_mask(
                            clip_model_dict["clip_model"],
                            clip_model_dict["clip_prs"],
                            clip_model_dict["clip_preprocess"],
                            "cuda",
                            clip_model_dict["clip_tokenizer"],
                            [image_pad_pil],
                            [cur_prompt]
                        )
                        mask = clip_merge_mask(cls_mask, patch_mask, kernel_size = args.kernel_size, enhance_coe = args.enhance_coe)
                        mask = toImg(mask.detach().cpu().unsqueeze(0))
                        interpolate_method = getattr(Image, "BICUBIC")
                        heatmap = invtrans(mask, image_pad_pil, method = interpolate_method)

                elif args.heatmap == 'api_llava':
                    attention_output = llava_hl.finalize()
                    llava_hl.reinit()
                    attention_output = attention_output.view(attention_output.shape[0],24,24)
                    attention_output = attention_output.detach()

                    input_token_len = input_ids.shape[1]
                    output_str = tokenizer.decode(output_ids[:, input_token_len:].cpu()[0])
                    token_mapping = get_token_mapping(tokenizer, output_str, output_ids[:, input_token_len:].cpu()[0])
                    state_cache = {"llava_attentions":attention_output.detach(), "llava_token_mapping":token_mapping}

                    mask = llava_from_preanswer_to_mask("", output_str, state_cache)

                    mask = revise_mask(mask.float(), kernel_size = args.kernel_size, enhance_coe = args.enhance_coe)
                    mask = mask.detach().cpu()
                    mask = toImg(mask.reshape(1,24,24))

                    interpolate_method = getattr(Image, "BICUBIC")
                    heatmap = invtrans(mask, image_pad_pil, method = interpolate_method)

                elif args.heatmap == 'relevancy_map':

                    with torch.enable_grad():

                        input_token_len = input_ids.shape[1]
                        img_idx=vis_start
                        generated_text = tokenizer.decode(output_ids[:, input_token_len:].cpu()[0])
                        generated_text_tokenized = tokenizer.tokenize(generated_text)
                        
                        word_rel_map = construct_relevancy_map(
                            tokenizer=tokenizer, 
                            model=model,
                            input_ids=input_ids,
                            tokens=generated_text_tokenized, 
                            outputs=model_outputs, 
                            output_ids=output_ids[:, input_token_len:].cpu()[0],
                            img_idx=vis_start
                        )
                        word_rel_map_ = word_rel_map['llama']

                        i = 0
                        for rel_key, rel_map in word_rel_map_.items():
                            i+=1
                            if rel_key in SEPARATORS_LIST:
                                continue
                            if (rel_map.shape[-1] != 577) and img_idx:
                                rel_map_out = rel_map[-1,:][img_idx:img_idx+576].reshape(24,24).detach().float().cpu().numpy()
                            else:
                                rel_map_out = rel_map[0,1:].reshape(24,24).detach().cpu().numpy()
                            del rel_map
                            break
                        keys_to_delete = [k for k in word_rel_map.keys()]
                        for k in keys_to_delete:
                            del word_rel_map[k]
                        del model.enc_attn_weights
                        model.enc_attn_weights = []

                    if args.normalize_mask == 'norm_enhance':
                        heatmap_th = revise_mask(torch.Tensor(rel_map_out), kernel_size=args.kernel_size, enhance_coe=args.enhance_coe)
                        heatmap = heatmap_th.detach().cpu()
                        mask = toImg(heatmap.reshape(1,24,24))
                        interpolate_method = getattr(Image, "BICUBIC") 
                        mask = invtrans(mask, image_pad_pil, method = interpolate_method)
                        heatmap = mask
                    else:
                        raise NotImplementedError

                if args.resample_method == 'magnify':
                    if isinstance(heatmap, torch.Tensor):
                        heatmap = heatmap.numpy()
                    elif isinstance(heatmap, Image.Image):
                        heatmap = np.array(heatmap)
                    height, width = pad_mask.shape[:2]
                    target_size = (height, width)
                    resampled_image = attention_sampling(np.array(image_pad_pil), heatmap, target_size, method = 'bilinear')
                    resampled_image = resampled_image.astype(np.uint8)
                    resampled_image = Image.fromarray(resampled_image, 'RGB')
                    

                elif args.resample_method == 'api':
                    if args.normalize_mask == 'regularization' and (args.heatmap == 'attention' or args.heatmap == 'iterative'):
                        raise Exception("Not really a good idea, reg term is not designed for api mask distribution")
                    if isinstance(heatmap, torch.Tensor) or isinstance(heatmap, np.ndarray):
                        heatmap = to_pil_image(heatmap, mode='L')
                    resampled_image = merge(heatmap.convert("L"), image_pad_pil.convert("RGB"), args.mask_grayscale).convert("RGB")
                elif args.resample_method == 'mag_api':
                    if isinstance(heatmap, torch.Tensor) or isinstance(heatmap, np.ndarray):
                        heatmap = to_pil_image(heatmap, mode='L')
                    resampled_image = merge(heatmap.convert("L"), image_pad_pil.convert("RGB"), args.mask_grayscale).convert("RGB")
                    if isinstance(heatmap, torch.Tensor):
                        heatmap = heatmap.numpy()
                    elif isinstance(heatmap, Image.Image):
                        heatmap = np.array(heatmap)
                    height, width = pad_mask.shape[:2]
                    target_size = (height, width)
                    resampled_image = attention_sampling(np.array(resampled_image), heatmap, target_size, method = 'bilinear')
                    resampled_image = resampled_image.astype(np.uint8)
                    resampled_image = Image.fromarray(resampled_image, 'RGB')
                elif args.resample_method == 'blurring':
                    if isinstance(heatmap, torch.Tensor) or isinstance(heatmap, np.ndarray):
                        heatmap = to_pil_image(heatmap, mode='L')
                    resampled_image = merge_with_blur(heatmap.convert("L"), image_pad_pil.convert("RGB"), args.blur_std_dev, args.mask_threshold).convert("RGB")
                elif args.resample_method == 'rectangle':
                    if isinstance(heatmap, torch.Tensor) or isinstance(heatmap, np.ndarray):
                        heatmap = to_pil_image(heatmap, mode='L')
                    resampled_image = merge_with_bbox(heatmap.convert("L"), image_pad_pil.convert("RGB"), args.mask_threshold, mute=True).convert("RGB")
                elif args.resample_method == 'crop':
                    if isinstance(heatmap, torch.Tensor) or isinstance(heatmap, np.ndarray):
                        heatmap = to_pil_image(heatmap, mode='L')
                    resampled_image = crop_with_mask(heatmap.convert("L"), image_pad_pil.convert("RGB"), args.mask_threshold).convert("RGB")

                if args.save_resampled_image:
                    if 'MME' in args.image_folder:
                        image_save_path = os.path.join(image_save_folder, f'{image_file}_{ith_batch}')
                    else:
                        image_save_path = os.path.join(image_save_folder, image_file)
                    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                    resampled_image.save(image_save_path, format="PNG")
                
                if args.heatmap == 'relevancy_map':
                    del model_outputs
                
                with torch.inference_mode():
                    resampled_image = process_images([resampled_image], image_processor, model.config)
                    args_dict['images'] = resampled_image.to(dtype=torch.float16, device='cuda', non_blocking=True)
                    model_outputs = model.generate(**args_dict)
                    output_ids = model_outputs['sequences']
                    scores = model_outputs['scores'][0]
                

        logits_raw = None
        if args.return_logits:
            tokens_naive, logits_raw = calibrate_label_dict(scores, tokenizer, return_logits=True)
            average_logis = scores[0].mean().item()
        else:
            tokens_naive = calibrate_label_dict(scores, tokenizer)
            
        if args.image_resample and args.resample_cd != 0:
            cd_alpha = args.resample_cd
            new_scores = scores + cd_alpha * (scores - original_scores)
            if args.return_logits:
                tokens_naive, logits_raw = calibrate_label_dict(new_scores, tokenizer, return_logits=True)
                average_logis = scores[0].mean().item()
            else:
                tokens_naive = calibrate_label_dict(new_scores, tokenizer)

        p_y = get_prob_from_logits(tokens_naive)


        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        if args.resample_during_sampling:
            del attention
            del model_outputs
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if args.heatmap == 'relevancy_map':
            del attention
            del model_outputs
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            del model.enc_attn_weights
            model.enc_attn_weights = []

        ith_batch += 1
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "logits_score": p_y,
                                   "image": image_file,
                                   "naive": tokens_naive,
                                   'logits_raw': logits_raw,
                                   'average_logits': average_logis if args.return_logits else None, 
                               
                                   "original_p_y": original_p_y,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data1/yifan/AIGC/datasets/MME_Benchmark")
    parser.add_argument("--question-file", type=str, default="./eval/MME/llava_mme.jsonl")
    parser.add_argument("--answers-file", type=str, default="./eval/MME/answers/llava-v1.5-7b-setting.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    # top_k will not only sample among the top-probs outcomes but also modify the final logits
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=10)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--return_cd_scores", action='store_true', default=False)
    parser.add_argument("--return_logits", action='store_true', default=False)
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--load_8bit", action='store_true', default=False)
    
    parser.add_argument("--image_resample", action='store_true', default=False, help="Whether to resample the image during the generation process")
    parser.add_argument("--resample_method",  type=str, default='magnify', help="Method to prompt the image, options: magnify, api, mag_api, blurring, rectangle, crop")
    parser.add_argument("--map_constructor", type=str, default='norm_average', help="How to construct the attention map from the raw attention values")
    parser.add_argument("--resample_regularization", type=float, default=0.3, help="Regularization term to add to the attention map, only used when map_constructor is regularization")
    parser.add_argument("--heatmap", type=str, default=None, help="which heatmap to use")
    parser.add_argument("--deep_layer", type=int, default=12, help="which layer to start aggregating the attention map")
    parser.add_argument("--exit_criteria", type=str, default='probability', help="Criteria to exit the iterative search, options: probability, sum_attention")
    parser.add_argument("--exit_threshold", type=float, default=0.1, help="Threshold to exit the iterative search, only used when exit_criteria is probability or sum_attention")
    parser.add_argument("--max_iteration", type=int, default=8, help="Maximum number of iterations for the iterative search")
    parser.add_argument("--iterative_search", type=str, default=None, help='whether to use iterative search, options: token_mask, None')
    parser.add_argument("--save_resampled_image", action='store_true', default=False, help="Whether to save the resampled image")
    parser.add_argument("--iterative_regularization", action='store_true', default=False, help="Whether to use regularization term for the iterative search")

    parser.add_argument("--normalize_mask", type=str, default='regularization', help="How to normalize the attention map, norm_enhance is the method used in paper")
    parser.add_argument("--enhance_coe", type=int, default = 5, help='alpha in paper')
    parser.add_argument("--kernel_size", type=int, default = 3, help='kernel size in paper')
    parser.add_argument("--mask_grayscale", type=int, default = 0)
    
    parser.add_argument("--resample_cd", type=float, default = 0, help='whether to apply contrastive decoding on the logits after resample')
    parser.add_argument("--aggreagte_api_attns", action='store_true', default=False)
    parser.add_argument("--api_llava_layer", type=int, default = 20)
    parser.add_argument("--single_layer", action='store_true', default=False, help='whether to use single layer attention instead of multiple layers')

    parser.add_argument("--blur_std_dev", type=int, default=100, 
                        help="standard deviation of Gaussian blur")
    parser.add_argument("--mask_threshold", type=float, default=0.5)

    parser.add_argument("--resample_during_sampling", action='store_true', default=False, help='resample during sampling function call, it should be used when multiple token generation is required') # For open end generation


    
    args = parser.parse_args()
    set_seed(args.seed)
    print(args)
    print(args.seed)

    import copy
    import numpy as np
    default_args = copy.deepcopy(args)
    answers_file = copy.deepcopy(args.answers_file)
    
    args.temperature = 1.0
    eval_model(args)
    args = default_args