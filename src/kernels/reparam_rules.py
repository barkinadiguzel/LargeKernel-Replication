import torch
import torch.nn.functional as F


def pad_kernel(kernel, target_size):
    k = kernel.size(-1)
    pad = (target_size - k) // 2
    return F.pad(kernel, [pad, pad, pad, pad])


def fuse_kernels(kernel_list, target_size):
    fused = 0
    for k in kernel_list:
        fused = fused + pad_kernel(k, target_size)
    return fused
