




from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import os

cache_dir = '/vast/users/schittyvenkata/model_weights/'
dataset = load_dataset("lmms-lab/MME", cache_dir = cache_dir, split = "test")

model_name = 'allenai/MolmoE-1B-0924'
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    cache_dir = cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    cache_dir = cache_dir
)

def process_sample(sample):
    
    image = sample['image']
    if image.mode != "RGB":
        image = image.convert("RGB")

    question = sample['question']
    text = question

    inputs = processor.process(
        images=[image],
        text=text,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    answer = sample['answer']

    encoding = processor.tokenizer(answer, return_tensors="pt")
    labels = encoding.input_ids.to(model.device)

    outputs = model(**inputs, labels=labels)

    loss = outputs.loss.item()
    print("loss = ", loss)
    return loss

total_loss = 0.0
num_samples = 0

for split_name in dataset:
    split = dataset[split_name]
    for sample in split:
        loss = process_sample(sample)
        if loss is not None:
            total_loss += loss
            num_samples += 1

if num_samples > 0:
    average_loss = total_loss / num_samples
    print(f"Average Loss: {average_loss}")
else:
    print("No samples were processed successfully.")




















# from PIL import Image
# import torch
# import random
# import requests

# from transformers import AutoModelForCausalLM, AutoProcessor
# from MolMoE.modeling_molmoe import MolmoForCausalLM
# from datasets import load_dataset

# cache_dir = '/vast/users/schittyvenkata/model_weights/'

# # dataset = load_dataset("lmms-lab/MME", cache_dir = cache_dir)
# dataset = load_dataset("lmms-lab/MME", cache_dir = cache_dir)['test']


# processor = AutoProcessor.from_pretrained(
#         'allenai/MolmoE-1B-0924',
#         trust_remote_code=True, cache_dir=cache_dir 
#     )
# model = AutoModelForCausalLM.from_pretrained(
#         'allenai/MolmoE-1B-0924',
#         trust_remote_code=True, 
#         torch_dtype='auto',
#         cache_dir=cache_dir
#     )

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # model = MolmoForCausalLM.from_pretrained(
# #     'allenai/MolmoE-1B-0924',
# #     trust_remote_code=True,
# #     torch_dtype='auto',
# #     # device_map='auto',
# #     cache_dir=cache_dir,
# # )

# # processor = AutoProcessor.from_pretrained(
# #     'allenai/MolmoE-1B-0924',
# #     trust_remote_code=True,
# #     cache_dir=cache_dir
# # )

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model.to(device)



# losses = []

# for example in dataset:
#     # Extract image, question, and answer
#     image = example['image']
#     question = example['question']
#     answer = example['answer']
    
#     inputs = processor.process(
#         images=[image],  # Wrap image in list
#         text=question
#     )
    
#     # Move tensors to device and add batch dimension
#     inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

#     labels = processor.tokenizer(
#         answer,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=32,  # Adjust based on model's capabilities
#         truncation=True
#     ).input_ids.to(device)




#     # # Process inputs (image + question)
#     # inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    
#     # # Tokenize answer for labels
#     # labels = processor.tokenizer(
#     #     answer,
#     #     return_tensors="pt",
#     #     padding="max_length",
#     #     max_length=32,  # Adjust based on model's max length
#     #     truncation=True
#     # ).input_ids.to(device)
    
#     # Replace padding tokens with -100 to ignore in loss
#     labels[labels == processor.tokenizer.pad_token_id] = -100
    
#     # Forward pass to compute loss
#     with torch.no_grad():
#         outputs = model(**inputs, labels=labels)
#         loss = outputs.loss.item()
    
#     losses.append(loss)
#     print(f"Loss: {loss}")













# # # def process_sample(sample):
    
# # #     image = sample["image"]
# # #     question = sample["question"]
# # #     answer = sample["answer"]

# # #     image_tensor = processor.feature_extractor(images=image, return_tensors="pt").to(device)
# # #     question_ids = processor.tokenizer(question, return_tensors="pt").to(device)

# # #     inputs = {
# # #         "pixel_values": image_tensor["pixel_values"],
# # #         "input_ids": question_ids["input_ids"],
# # #         "attention_mask": question_ids["attention_mask"],

# # #     }


# # #     try:
# # #         with torch.no_grad():
# # #             outputs = model(**inputs)
# # #             logits = outputs.logits
# # #             loss = outputs.loss
# # #     except Exception as e:
# # #         print(f"Error processing sample: {e}")
# # #         return None, None

# # #     return logits, loss



# # def process_sample(sample):
# #     image = sample["image"]
# #     question = sample["question"]
# #     answer = sample["answer"]

# #     inputs = processor(images=image, text=question, return_tensors="pt").to(device)

# #     inputs = processor(images=image, text=question, return_tensors="pt").to(device)
# #     labels = processor(text=answer, return_tensors="pt", padding=True).to(device)
    
# #     with torch.no_grad():
# #         outputs = model(**inputs, labels=labels)
# #         logits = outputs.logits
# #         loss = outputs.loss

# #     return logits, loss

# # for i, sample in enumerate(dataset["test"]):
# #     print(f"Processing sample {i + 1}/{len(dataset['test'])}")

# #     logits, loss = process_sample(sample)

# #     if logits is not None and loss is not None:
# #         print(f"  Logits shape: {logits.shape}")
# #         print(f"  Loss: {loss.item()}")
# #     else:
# #         print("  Skipping sample due to processing error.")
