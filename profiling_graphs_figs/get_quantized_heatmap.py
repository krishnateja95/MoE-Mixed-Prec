

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def plot_heatmap(layer_dict, my_list, filename="frequency.pdf"):
    num_layers = len(layer_dict)
    num_sublayers = len(next(iter(layer_dict.values())))
    data = np.zeros((num_sublayers, num_layers))

    for layer_idx, sublayer_dict in layer_dict.items():
        for sublayer_idx, value in sublayer_dict.items():
            data[sublayer_idx, layer_idx] = value

    with PdfPages(filename) as pdf:
        plt.rcParams['font.family'] = 'serif'
        fig, ax = plt.subplots(figsize=(16, 5))  
        im = ax.imshow(data, cmap="Reds", aspect="auto")  

        title_fontsize = 20
        label_fontsize = 20
        tick_fontsize = 20

        ax.set_title("Mixed Precision Quantization Map", fontsize=title_fontsize)
        ax.set_xlabel("Layer Index", fontsize=label_fontsize)
        ax.set_ylabel("Expert Index", fontsize=label_fontsize)

        ax.set_xticks(np.arange(0, num_layers, 3)) 
        ax.set_yticks(np.arange(num_sublayers))
        ax.set_xticklabels(np.arange(0, num_layers, 3), fontsize=tick_fontsize)
        ax.set_yticklabels(np.arange(num_sublayers), fontsize=tick_fontsize)


        cbar = fig.colorbar(im, ax=ax, label='Bit Width', ticks=my_list)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label('Bit Width', size=tick_fontsize)

        plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

        fig.tight_layout()

        pdf.savefig(fig)

        plt.close(fig)

if __name__ == '__main__':
    import random

    num_layers = 32
    num_sublayers = 8
    my_list = [1, 2, 4, 8]


    layer_dict = {}
    for layer_index in range(num_layers):
        print(layer_index)
        
        expert_dict = {}
        for expert_index in range(num_sublayers):
            expert_dict[expert_index] = random.choice(my_list)

        layer_dict[layer_index] = expert_dict

    plot_heatmap(layer_dict, my_list, "quantized_experts.pdf")
    print("Heatmap saved to quantized_experts.pdf")




# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.backends.backend_pdf import PdfPages

# # Sample data (replace with your actual layer_dict)
# # Assuming layer_dict[layer_num][sublayer_num] = value
# layer_dict = {
#     i: {j: np.random.randint(0, 100) for j in range(8)}  # Example values
#     for i in range(32)
# }


# # Prepare data for heatmap
# heatmap_data = np.zeros((8, 32))  # 8 sublayers, 32 layers
# for layer_num in range(32):
#     for sublayer_num in range(8):
#         heatmap_data[sublayer_num, layer_num] = layer_dict[layer_num][sublayer_num]


# # Plotting
# def plot_heatmap(data, filename="frequency.pdf", title="Activation Frequency Map",
#                  xlabel="Layers", ylabel="Sublayers",
#                  title_fontsize=16, label_fontsize=12, tick_fontsize=10):
#     """
#     Generates a heatmap plot and saves it as a PDF.

#     Args:
#         data (np.ndarray): 2D array of data for the heatmap.
#         filename (str): Name of the PDF file to save.
#         title (str): Title of the plot.
#         xlabel (str): Label for the x-axis.
#         ylabel (str): Label for the y-axis.
#         title_fontsize (int): Font size for the title.
#         label_fontsize (int): Font size for the axis labels.
#         tick_fontsize (int): Font size for the axis ticks.
#     """
#     with PdfPages(filename) as pdf:
#         fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size as needed

#         # Create the heatmap
#         im = ax.imshow(data, cmap="Blues", aspect="auto")  # You can change the colormap

#         # Set labels and title
#         ax.set_title(title, fontsize=title_fontsize)
#         ax.set_xlabel(xlabel, fontsize=label_fontsize)
#         ax.set_ylabel(ylabel, fontsize=label_fontsize)

#         # Set ticks
#         ax.set_xticks(np.arange(0, data.shape[1], 2))
#         ax.set_yticks(np.arange(data.shape[0]))

#         # Label ticks
#         ax.set_xticklabels(np.arange(data.shape[1]), fontsize=tick_fontsize)
#         ax.set_yticklabels(np.arange(data.shape[0]), fontsize=tick_fontsize)

#         # Rotate x-axis labels for better readability (optional)
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#         # Add colorbar
#         cbar = ax.figure.colorbar(im, ax=ax)
#         cbar.ax.tick_params(labelsize=tick_fontsize)  # Adjust colorbar tick size

#         # Add text annotations inside the heatmap (optional)
#         # for i in range(data.shape[0]):
#         #     for j in range(data.shape[1]):
#         #         text = ax.text(j, i, data[i, j],
#         #                        ha="center", va="center", color="w") # Choose color based on cmap

#         fig.tight_layout()  # Adjust layout to prevent labels from overlapping
#         pdf.savefig(fig)
#         plt.close(fig)
#         print(f"Heatmap saved to {filename}")


# # Example usage
# fontsize = 20
# plot_heatmap(heatmap_data,
#              filename="frequency.pdf",
#              title="Activation Frequency Map",
#              xlabel="Layers",
#              ylabel="Sublayers",
#              title_fontsize=fontsize,
#              label_fontsize=fontsize,
#              tick_fontsize=fontsize)


















# # import matplotlib.pyplot as plt
# # import numpy as np
# # import matplotlib.backends.backend_pdf

# # def plot_heatmap(layer_dict, fontsize = 20, output_filename="frequency.pdf"):
# #     num_layers = len(layer_dict)
# #     num_sublayers = 8

# #     heatmap_data = np.zeros((num_sublayers, num_layers))

# #     for layer_idx in layer_dict:
# #         for sublayer_idx in range(num_sublayers):
# #             heatmap_data[sublayer_idx, layer_idx] = layer_dict[layer_idx][sublayer_idx]

# #     plt.rcParams['font.family'] = 'serif'
# #     plt.xticks(fontsize=fontsize)
# #     plt.yticks(fontsize=fontsize)

# #     fig, ax = plt.subplots(figsize=(16, 6))

# #     im = ax.imshow(heatmap_data, cmap="Blues", aspect="auto") 

    
# #     ax.set_title("Activation Frequency Map", fontsize = fontsize)
# #     ax.set_xlabel("Layer Index", fontsize = fontsize) # optional
# #     ax.set_ylabel("Expert Index", fontsize = fontsize) # optional


# #     # Save the plot to a PDF file
# #     pdf = matplotlib.backends.backend_pdf.PdfPages(output_filename)
# #     pdf.savefig(fig)
# #     pdf.close()

# #     print(f"Heatmap saved to {output_filename}")


# # if __name__ == '__main__':
# #     # Example Usage (replace with your actual layer_dict)
# #     # Generate a sample layer_dict for demonstration:
# #     num_layers = 32
# #     layer_dict = {}
# #     for i in range(num_layers):
# #         layer_dict[i] = {}
# #         for j in range(8):
# #             layer_dict[i][j] = np.random.rand()  # Replace with your actual values

# #     plot_heatmap(layer_dict)
