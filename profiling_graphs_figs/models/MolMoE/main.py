
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

# cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

cache_dir = '/vast/users/schittyvenkata/model_weights/'

processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    cache_dir=cache_dir
)

from modeling_molmoe import MolmoForCausalLM
import random

model = MolmoForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    cache_dir=cache_dir,
)



inputs = processor.process(images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
                           text="Describe this image.")

inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

output = model.generate_from_batch(inputs,
                                   GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                                   tokenizer=processor.tokenizer)

generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(generated_text)

