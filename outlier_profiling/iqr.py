



import torch
import numpy as np
import torch.nn as nn

# Initialize a tensor with 8 random weight values
network_layer = nn.Linear(4096, 4096)
weights_tensor =  network_layer.weight # Normally distributed random weights



weights = weights_tensor.detach().numpy()  # Convert to NumPy array for easier processing

# IQR Method
Q1 = np.percentile(weights, 25)
Q3 = np.percentile(weights, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = weights[(weights < lower_bound) | (weights > upper_bound)]

# Z-Score Method
mean = np.mean(weights)
std_dev = np.std(weights)
z_scores = (weights - mean) / std_dev
outliers_z = weights[np.abs(z_scores) > 3]

# Print results
print(f"Initialized Weights: {weights}")
print(f"Outliers (IQR method): {outliers_iqr}")
print(f"Outliers (Z-score method): {outliers_z}")










# import torch
# import torch.nn as nn

# def count_outliers_iqr(tensor):
#     q1 = tensor.quantile(0.25)
#     q3 = tensor.quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     outliers = ((tensor < lower_bound) | (tensor > upper_bound)).sum().item()
#     return outliers

# num_experts = 8
# layers_per_expert = 3
# input_dim = 128
# hidden_dim = 256
# output_dim = 128

# experts = {}

# for i in range(num_experts):
#     expert = nn.Sequential(
#         nn.Linear(input_dim, hidden_dim),
#         nn.Linear(hidden_dim, hidden_dim),
#         nn.Linear(hidden_dim, output_dim)
#     )
#     experts[f"expert_{i}"] = expert

# outlier_counts = {}
# for name, expert in experts.items():
#     total_outliers = sum(count_outliers_iqr(param) for param in expert.parameters())
#     outlier_counts[name] = total_outliers

# most_outliers_expert = max(outlier_counts, key=outlier_counts.get)
# most_outliers_count = outlier_counts[most_outliers_expert]

# print(f"Expert with most outliers: {most_outliers_expert}")
# print(f"Number of outliers: {most_outliers_count}")
