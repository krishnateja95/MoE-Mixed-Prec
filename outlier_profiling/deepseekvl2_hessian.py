
from DeepSeek_VL2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
import torch
from transformers import AutoConfig
from Hutchinson_estimate import hessian_trace_hutchinson
import numpy as np


deepseekvl2_series = {
    'deepseek_vl2_tiny': 'deepseek-ai/deepseek-vl2-tiny',
    'deepseek_vl2_small': 'deepseek-ai/deepseek-vl2-small',
    'deepseek_vl2': 'deepseek-ai/deepseek-vl2',
}



deepseekvl2_num_layers = {
    'deepseek_vl2_tiny': 12,
    'deepseek_vl2_small': 27,
    'deepseek_vl2': 30,
}


deepseekvl2_num_experts = {
    'deepseek_vl2_tiny': 64,
    'deepseek_vl2_small': 64,
    'deepseek_vl2': 64,
}




def split_model(model_name):
    if model_name == 'deepseek-ai/deepseek-vl2-tiny':
        return "cuda:0"
    
    device_map = {}
    model_splits = {
        'deepseek-ai/deepseek-vl2-small': [13, 14], # 2 GPU for 16b
        'deepseek-ai/deepseek-vl2': [10, 10, 10], # 3 GPU for 27b
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers =  sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map


for model_name in [
    'deepseek_vl2_tiny',
    # 'deepseek_vl2_small',
    # 'deepseek_vl2'
    ]:

    model_path = deepseekvl2_series[model_name]
    device_map = split_model(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = DeepseekVLV2ForCausalLM.from_pretrained(model_path,
                                                    trust_remote_code=True,
                                                    torch_dtype= torch.bfloat16,
                                                    device_map = device_map,
                                                    cache_dir  = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/')
    
    hessian_dict = {}

    for layer_id in range(1, deepseekvl2_num_layers[model_name]):
        for expert_id in range(0, deepseekvl2_num_experts[model_name]):

            print(f"Layer = {layer_id} and Expert = {expert_id}")
            
            for name, param in model.named_parameters():
                if name == f"language.model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight":
                    gate_proj_weight = param
                
                if name == f"language.model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight":
                    up_proj_weight = param

                if name == f"language.model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight":
                    down_proj_weight = param

            gate_hessian = hessian_trace_hutchinson(gate_proj_weight)
            up_hessian   = hessian_trace_hutchinson(up_proj_weight)
            down_hessian = hessian_trace_hutchinson(down_proj_weight) 

            hessian_dict[f"language.model.layers.{layer_id}.mlp.experts.{expert_id}"] = gate_hessian + up_hessian + down_hessian

    # for layer_id in range(1, deepseekvl2_num_layers[model_name]):
    #     layer_experts = [hessian_dict[f"language.model.layers.{layer_id}.mlp.experts.{expert_id}"] for expert_id in range(deepseekvl2_num_experts[model_name])]
    #     sum_layer = np.mean(layer_experts)
    
    #     for expert_id in range(0, deepseekvl2_num_experts[model_name]):
    #         key = f"language.model.layers.{layer_id}.mlp.experts.{expert_id}"
    #         hessian_dict[key] /= sum_layer

    print(hessian_dict)
    

