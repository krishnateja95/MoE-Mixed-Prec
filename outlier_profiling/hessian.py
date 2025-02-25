import torch
import torch.nn as nn

def compute_hessian_approx(weights):
    """
    Compute an approximation of the diagonal of the Hessian using 
    the empirical Fisher Information Matrix (FIM).
    """
    grad_squared = torch.pow(torch.autograd.functional.jacobian(lambda w: torch.norm(w), weights), 2)
    hessian_diag = grad_squared.mean(dim=tuple(range(1, grad_squared.dim())))  # Aggregate across weight dimensions
    return hessian_diag

# Define example layers with random weights
layer1 = nn.Linear(4096, 4096)
layer2 = nn.Linear(4096, 4096)
layer3 = nn.Linear(4096, 4096)
layer4 = nn.Linear(4096, 4096)
layer5 = nn.Linear(4096, 4096)

# Extract weights
weights1 = layer1.weight
weights2 = layer2.weight
weights3 = layer3.weight
weights4 = layer4.weight
weights5 = layer5.weight

# Compute Hessian approximations
hessian1 = compute_hessian_approx(weights1)
hessian2 = compute_hessian_approx(weights2)
hessian3 = compute_hessian_approx(weights3)
hessian4 = compute_hessian_approx(weights4)
hessian5 = compute_hessian_approx(weights5)


# hess_avg = (hessian1 + hessian2 + hessian3 + hessian4 + hessian5)/5

print("Layer 1 Sensitivity:", hessian1)


# Print sensitivity results
print("Layer 1 Sensitivity:", hessian1.mean().item())
# print("Layer 2 Sensitivity:", hessian2.mean().item())
# print("Layer 3 Sensitivity:", hessian3.mean().item())
# print("Layer 4 Sensitivity:", hessian4.mean().item())
# print("Layer 5 Sensitivity:", hessian5.mean().item())
# print("Layer 5 Sensitivity:", hessian5.mean().item())
