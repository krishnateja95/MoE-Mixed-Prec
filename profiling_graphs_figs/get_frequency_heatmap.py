

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def plot_heatmap(layer_dict, filename="frequency.pdf"):
    num_layers = len(layer_dict)
    num_sublayers = len(next(iter(layer_dict.values())))
    data = np.zeros((num_sublayers, num_layers))

    for layer_idx, sublayer_dict in layer_dict.items():
        for sublayer_idx, value in sublayer_dict.items():
            data[sublayer_idx, layer_idx] = value

    with PdfPages(filename) as pdf:
        plt.rcParams['font.family'] = 'serif'
        fig, ax = plt.subplots(figsize=(16, 5))  
        im = ax.imshow(data, cmap="Blues", aspect="auto")  

        title_fontsize = 20
        label_fontsize = 20
        tick_fontsize = 20

        ax.set_title("Activation Frequency Map", fontsize=title_fontsize)
        ax.set_xlabel("Layer Index", fontsize=label_fontsize)
        ax.set_ylabel("Expert Index", fontsize=label_fontsize)

        ax.set_xticks(np.arange(0, num_layers, 3)) 
        ax.set_yticks(np.arange(num_sublayers))
        ax.set_xticklabels(np.arange(0, num_layers, 3), fontsize=tick_fontsize)
        ax.set_yticklabels(np.arange(num_sublayers), fontsize=tick_fontsize)


        cbar = fig.colorbar(im, ax=ax, label='Frequency')
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label('Frequency', size=tick_fontsize)

        plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

        fig.tight_layout()

        pdf.savefig(fig)

        plt.close(fig)

if __name__ == '__main__':
    import random

    num_layers = 32
    num_sublayers = 8

    layer_dict = {}
    for layer_index in range(num_layers):
        print(layer_index)
        
        expert_dict = {}
        for expert_index in range(num_sublayers):
            expert_dict[expert_index] = random.randint(1, 10)

        layer_dict[layer_index] = expert_dict

    plot_heatmap(layer_dict, "frequency.pdf")
    print("Heatmap saved to frequency.pdf")


