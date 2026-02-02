import matplotlib.pyplot as plt
import torch


def visualize_kernel(kernel):
    k = kernel.squeeze().detach().cpu()
    plt.imshow(k, cmap="viridis")
    plt.colorbar()
    plt.show()


def visualize_fused_kernel(branches, fused):
    n = len(branches) + 1
    plt.figure(figsize=(3 * n, 3))

    for i, b in enumerate(branches):
        plt.subplot(1, n, i + 1)
        visualize_kernel(b.conv.weight[0])

    plt.subplot(1, n, n)
    visualize_kernel(fused[0])
