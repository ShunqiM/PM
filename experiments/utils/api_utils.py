import sys
import os

# Get the current working directory and add 'experiments' to the path
sys.path.insert(0, os.path.join(os.getcwd(), 'experiments'))

import torch
from PIL import Image
import matplotlib.pyplot as plt
from experiments.API.API_LLaVA.functions import get_model as llava_get_model, get_preanswer as llava_get_preanswer, from_preanswer_to_mask as llava_from_preanswer_to_mask
from experiments.API.API_LLaVA.main import blend_mask as llava_blend_mask
from experiments.API.API_CLIP.main import get_model as clip_get_model, gen_mask as clip_gen_mask, blend_mask as clip_blend_mask
from experiments.API.API_LLaVA.hook import hook_logger as llava_hook_logger
from experiments.llava.model.language_model.llava_llama import llama_modify
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CLIP model
def init_clip():
    clip_model, clip_prs, clip_preprocess, _, clip_tokenizer = clip_get_model(
        model_name="ViT-L-14-336" if torch.cuda.is_available() else "ViT-L-14",
        layer_index=22,
        device=DEVICE
    )
    print(clip_model)
    return {"clip_model": clip_model, "clip_prs": clip_prs, "clip_preprocess": clip_preprocess, "clip_tokenizer": clip_tokenizer}

# Initialize LLaVA model
def init_llava():
    llava_tokenizer, llava_model, llava_image_processor, llava_context_len, llava_model_name = llava_get_model("llava-v1.5-7b", device=DEVICE)
    llava_hl = llava_hook_logger(llava_model, DEVICE, layer_index = 20, single_layer=False)
    llama_modify(llava_model, 20, 32, False, 0, False, 35, 35 + 576)
    return {"llava_tokenizer": llava_tokenizer, "llava_model": llava_model, "llava_image_processor": llava_image_processor, "llava_context_len": llava_context_len, "llava_model_name": llava_model_name, "llava_hl": llava_hl}

# Generate masked image using CLIP or LLaVA
def get_masked_image(image, question, highlight_text, model_choice="CLIP", enhance_coe=5, kernel_size=3, interpolate_method="LANCZOS", mask_grayscale=100):
    if model_choice == "CLIP":
        model_dict = init_clip()
        # Generate mask using CLIP model
        with torch.no_grad():
            clip_mask = clip_gen_mask(
                model_dict["clip_model"],
                model_dict["clip_prs"],
                model_dict["clip_preprocess"],
                DEVICE,
                model_dict["clip_tokenizer"],
                [image],
                [highlight_text if highlight_text.strip() != "" else question]
            )
        
        masked_image = clip_blend_mask(image, *clip_mask, enhance_coe, kernel_size, interpolate_method, mask_grayscale)
        
    elif model_choice == "LLaVA":
        model_dict = init_llava()
        # Get pre-answer and generate mask using LLaVA model
        pre_answer, state_cache = llava_get_preanswer(
            model_dict["llava_model"],
            model_dict["llava_model_name"],
            model_dict["llava_hl"], 
            model_dict["llava_tokenizer"],
            model_dict["llava_image_processor"],
            model_dict["llava_context_len"],
            question,
            image
        )
        print(pre_answer)
        mask = llava_from_preanswer_to_mask(highlight_text, pre_answer, state_cache)
        print(mask.shape)
        masked_image = llava_blend_mask(image, mask, enhance_coe, kernel_size, interpolate_method, mask_grayscale)
        
    else:
        raise ValueError("Invalid model choice! Use 'CLIP' or 'LLaVA'.")

    return masked_image


from llava.conversation import conv_templates, SeparatorStyle
if __name__ == '__main__':
    # Load image and define question and highlight text
    image_path = "./sampler_test.png"
    image = Image.open(image_path).convert("RGB")
    question = "Is there a glass in the image? Please answer this question with one word."
    highlight_text = ""  # Adjust based on pre-answer or query content
    model_choice = "CLIP"  # Change to "LLaVA" if you want to use LLaVA model
    model_choice = "LLaVA"  

    # Generate masked image
    masked_image = get_masked_image(image, question, highlight_text, model_choice=model_choice)

    # Display the masked image
    plt.imshow(masked_image)
    plt.axis("off")
    plt.show()