from llava.model.language_model.llava_stablelm import LlavaStableLmForCausalLM, LlavaStableLmModel, StableLmLlavaConfig
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import torch
import os
config = StableLmLlavaConfig.from_pretrained("/weka/home-reshinth/LLaVA/checkpoints/llava-v1.5-stablelm-3b-pretrain")
model = LlavaStableLmForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b", config=config)
model_old = model
mm_projector_weights = torch.load(os.path.join("/weka/home-reshinth/LLaVA/checkpoints/llava-v1.5-stablelm-3b-pretrain", 'mm_projector.bin'), map_location='cpu')
mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
print(mm_projector_weights)
model.load_state_dict(mm_projector_weights, strict=False)
print(model == model_old)
print(model.state_dict().keys())
