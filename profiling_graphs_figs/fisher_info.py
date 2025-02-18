import torch
import torch.nn as nn

# Example: Calculating Fisher Information for a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(5, 3)  # Example layer

    def forward(self, x):
        return self.fc(x)

# Initialize model and data
model = SimpleModel()
data = torch.randn(10, 5)  # Example data

# Calculate gradients (simplified for illustration)
def calculate_fisher_info(model, data):
    # Compute log-likelihood gradients
    outputs = model(data)
    loss = nn.MSELoss()(outputs, torch.randn_like(outputs))  # Example loss
    gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    
    # Simplified Fisher info calculation (actual implementation requires more steps)
    fisher_info = [grad.pow(2).mean() for grad in gradients]
    return fisher_info

# Calculate Fisher information
fisher_info = calculate_fisher_info(model, data)

# Use Fisher information to determine weight sensitivity
sensitivity = [info.item() for info in fisher_info]
print(sensitivity)
