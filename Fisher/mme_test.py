import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO


cache_dir = '/vast/users/schittyvenkata/model_weights/'
processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    # device_map='auto',
    cache_dir = cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    cache_dir = cache_dir
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


dataset = load_dataset('lmms-lab/MME', split='test', cache_dir = cache_dir)


def process_sample(sample):
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    inputs = processor.process(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    labels = processor.tokenizer(text=answer, return_tensors='pt')['input_ids'].to(model.device)
    return inputs, labels


model.train()

for sample in dataset:
    inputs, labels = process_sample(sample)

    outputs = model(**inputs)
    print(outputs)
