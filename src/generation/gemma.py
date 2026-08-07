from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from src.core.config import GENERAL_MODEL, HF_TOKEN

def loading_model(): 
    model_id = GENERAL_MODEL 
    # AutoModelForImageTextToText.from_pretrainted - Neu dung gemma4 
    vision_model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
    ).eval()
    vision_processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    return vision_model, vision_processor 

