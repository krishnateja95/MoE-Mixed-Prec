
import torch
from MolmoE.modeling_molmoe import MolmoForCausalLM
from Hutchinson_estimate import hessian_trace_hutchinson
from transformers import AutoConfig
import numpy as np

model_dict = {'molmoE-1B-0924': 'allenai/MolmoE-1B-0924'}

for model in ["molmoE-1B-0924"]:
    model_path = model_dict[model]
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model  = MolmoForCausalLM.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              torch_dtype=torch.bfloat16,
                                              device_map = "cuda:0",
                                              cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/')
    
    print("moe_num_experts", config.moe_num_experts)
    print("n_layers", config.n_layers)

    hessian_dict = {}

    for layer_id in range(0, config.n_layers):
        for expert_id in range(0, config.moe_num_experts):
            print(f"Layer = {layer_id} and Expert = {expert_id}")
            
            for name, param in model.named_parameters():
                if name == f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight":
                    gate_proj_weight = param
                
                if name == f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}.up_proj.weight":
                    up_proj_weight = param

                if name == f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}.down_proj.weight":
                    down_proj_weight = param

            gate_hessian = hessian_trace_hutchinson(gate_proj_weight)
            up_hessian   = hessian_trace_hutchinson(up_proj_weight)
            down_hessian = hessian_trace_hutchinson(down_proj_weight) 

            hessian_dict[f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}"] = gate_hessian + up_hessian + down_hessian


    # for layer_id in range(0, config.n_layers):
    #     avg_layer = 0
    #     for expert_id in range(0, config.moe_num_experts):
    #         avg_layer += hessian_dict[f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}"]
        
    #     avg_layer = avg_layer/config.moe_num_experts

    #     for expert_id in range(0, config.moe_num_experts):
    #         hessian_dict[f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}"] = hessian_dict[f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}"] / avg_layer 

    for layer_id in range(config.n_layers):
        layer_experts = [hessian_dict[f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}"] for expert_id in range(config.moe_num_experts)]
        sum_layer = np.sum(layer_experts)
    
        for expert_id in range(config.moe_num_experts):
            key = f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}"
            hessian_dict[key] /= sum_layer

    print(hessian_dict)




        

