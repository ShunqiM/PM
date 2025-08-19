import os, sys

sys.path.insert(0, os.path.join(os.getcwd(), 'experiments', 'API', 'API_CLIP/clip_prs'))
from experiments.API.API_CLIP.clip_prs.utils.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from experiments.API.API_CLIP.clip_prs.utils.factory import create_model, create_model_and_transforms, create_model_from_pretrained, get_tokenizer, create_loss
from experiments.API.API_CLIP.clip_prs.utils.factory import list_models, add_model_config, get_model_config, load_checkpoint
from experiments.API.API_CLIP.clip_prs.utils.pretrained import list_pretrained, list_pretrained_models_by_tag, list_pretrained_tags_by_model, \
    get_pretrained_url, download_pretrained_from_url, is_pretrained_cfg, get_pretrained_cfg, download_pretrained
from experiments.API.API_CLIP.clip_prs.utils.tokenizer import SimpleTokenizer, tokenize, decode
from experiments.API.API_CLIP.clip_prs.utils.transform import image_transform, AugmentationCfg
from experiments.API.API_CLIP.clip_prs.utils.openai_templates import OPENAI_IMAGENET_TEMPLATES