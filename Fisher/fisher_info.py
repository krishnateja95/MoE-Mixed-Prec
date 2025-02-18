from MolMoE.modeling_molmoe import MolmoeSparseMoeBlock
from MolMoE.modeling_molmoe import MolmoForCausalLM
import random
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

def compute_fisher_scores(model, calibration_data, device):
    # Initialize Fisher scores for all MoE layers
    for name, module in model.named_modules():
        if isinstance(module, MolmoeSparseMoeBlock):
            module.fisher_scores.zero_()

    # Run calibration
    model.eval()
    for batch in calibration_data:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        # Process each MoE layer
        for name, module in model.named_modules():
            if isinstance(module, MolmoeSparseMoeBlock):
                selected_experts = outputs.selected_experts  # Assuming outputs include selected_experts per layer
                expert_counts = module.expert_counts

                for expert_idx in range(module.num_experts):
                    expert = module.experts[expert_idx]
                    count = expert_counts[expert_idx].item()
                    if count == 0:
                        continue

                    # Sum squared gradients for expert's parameters
                    total_sq_grad = 0.0
                    for param in expert.parameters():
                        if param.grad is not None:
                            grad = param.grad
                            total_sq_grad += (grad ** 2).sum().item()

                    # Accumulate Fisher score (average per usage)
                    module.fisher_scores[expert_idx] += total_sq_grad / max(count, 1)

        model.zero_grad()

    # Normalize scores by number of calibration batches (optional)
    for name, module in model.named_modules():
        if isinstance(module, MolmoeSparseMoeBlock):
            module.fisher_scores /= len(calibration_data)
    
    return model


if __name__ == "__main__":

    cache_dir = '/vast/users/schittyvenkata/model_weights/'
    processor = AutoProcessor.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
        cache_dir=cache_dir
    )

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



