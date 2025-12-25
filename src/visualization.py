import matplotlib.pyplot as plt
import numpy as np


def plot_similarity_matrix(matrix, labels):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix)
    plt.colorbar(cax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.show()
