import torch
import torch.nn as nn

def hessian_trace_hutchinson(layer_weights, num_samples=100):
    """
    Approximate the trace of the Hessian using Hutchinson's method.
    
    Parameters:
    - model: Neural network model
    - layer_weights: Weight tensor of a specific layer
    - num_samples: Number of random samples for estimation
    
    Returns:
    - Estimated trace of the Hessian for the layer
    """
    trace_estimates = []
    
    for _ in range(num_samples):
        v = torch.randn_like(layer_weights).to(layer_weights.device)  # Sample random vector (Gaussian)
        v = v / torch.norm(v)  # Normalize for stability
        
        # Compute Hessian-vector product using second-order gradients
        layer_weights.requires_grad_(True)
        loss = torch.norm(layer_weights)  # Use weight norm as a proxy for loss
        grad1 = torch.autograd.grad(loss, layer_weights, create_graph=True)[0]
        hvp = torch.autograd.grad(grad1, layer_weights, grad_outputs=v)[0]
        
        trace_estimates.append(torch.sum(v * hvp).item())  # Hutchinson Estimation
    
    return sum(trace_estimates) / num_samples  # Average estimate

# # Define example layers
# layer1 = nn.Linear(4096, 4096)
# layer2 = nn.Linear(4096, 4096)
# layer3 = nn.Linear(4096, 4096)

# # Compute Hessian trace approximation for each layer
# trace1 = hessian_trace_hutchinson(layer1.weight)
# trace2 = hessian_trace_hutchinson(layer2.weight)
# trace3 = hessian_trace_hutchinson(layer3.weight)

# # Print estimated sensitivity
# print(f"Layer 1 (Linear 128x64) Hessian Trace: {trace1}")
# print(f"Layer 2 (Conv2D 16->32) Hessian Trace: {trace2}")
# print(f"Layer 3 (Linear 64x10) Hessian Trace: {trace3}")
