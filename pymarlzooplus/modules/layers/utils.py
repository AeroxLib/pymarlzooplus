# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 18:34
# @Author  : Zhuohui Zhang
# @File    : utils.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch

# 尝试导入 torch_scatter，如果失败则使用替代实现
try:
    import torch_scatter
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    print("Warning: torch_scatter not available, using fallback implementation")


def transpose_input(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x, num_heads):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


def sequence_mask(x, mask, value):
    mask = mask.to(torch.bool)
    x[:, 0][~mask] = value
    return x


def normalization(adjacency):
    tensor_adjacency = []
    for i in range(adjacency.size(0)):
        A_i = adjacency[i]
        tensor_adjacency_i = []
        for j in range(A_i.size(0)):
            A_ij = A_i[j].clone()
            A_ij += torch.eye(A_ij.shape[0], device=A_ij.device)
            degree = A_ij.sum(-1, keepdim=True)
            d_hat = torch.diag(torch.pow(degree, -0.5).flatten())
            tensor_adjacency_ij = torch.matmul(torch.matmul(d_hat, A_ij), d_hat)
            tensor_adjacency_i.append(tensor_adjacency_ij)
        tensor_adjacency.append(torch.stack(tensor_adjacency_i))
    tensor_adjacency = torch.stack(tensor_adjacency)

    return tensor_adjacency


def global_avg_pool(x, graph_indicator):
    """全局平均池化"""
    if HAS_TORCH_SCATTER:
        num = graph_indicator.max().item() + 1
        return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)
    else:
        # 替代实现
        num = graph_indicator.max().item() + 1
        result = []
        for i in range(num):
            mask = (graph_indicator == i)
            if mask.any():
                result.append(x[mask].mean(dim=0))
            else:
                result.append(torch.zeros_like(x[0]))
        return torch.stack(result)


def global_max_pool(x, graph_indicator):
    """全局最大池化"""
    if HAS_TORCH_SCATTER:
        num = graph_indicator.max().item() + 1
        return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]
    else:
        # 替代实现
        num = graph_indicator.max().item() + 1
        result = []
        for i in range(num):
            mask = (graph_indicator == i)
            if mask.any():
                result.append(x[mask].max(dim=0)[0])
            else:
                result.append(torch.zeros_like(x[0]))
        return torch.stack(result)


def corrupt(x, amount):
    noise = torch.randn_like(x)
    return x * (1 - amount) + noise * amount
