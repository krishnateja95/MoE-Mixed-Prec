


total_size_bits = sum(param.numel() * bits_dict[name] for name, param in model.named_parameters())
total_size_gb = total_size_bits / (8 * 10**9)
print(f"Total model size: {total_size_gb:.4f} GB")
