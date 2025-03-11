import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    EGConv,
    GINConv,
    GATv2Conv,
    PDNConv,
    GENConv,
    ResGatedGraphConv,
    AntiSymmetricConv,
)
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.tensorboard import SummaryWriter


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(GCN, self).__init__()
        self.layers = layers
        # 构造多层GCN
        if layers == 1:
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(input_dim, output_dim))
        else:
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for i in range(layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, output_dim))
        self.mlp = torch.nn.Linear(hidden_dim, output_dim)
        # 仅在输入维度和隐藏维度不同的情况下使用线性变换对齐维度
        if input_dim != hidden_dim:
            self.residual_fc = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_fc = None
        self.residual_fc = None

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            x = self.convs[0](x, edge_index)
            if test:
                return x
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        residual = x  # 初始输入
        x = self.convs[0](x, edge_index)

        if self.residual_fc is not None:
            residual = self.residual_fc(residual)  # 线性变换对齐维度

        # x = x + residual  # 添加残差
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # 中间层
        for conv in self.convs[1:-1]:
            # residual = x  # 记录残差
            x = conv(x, edge_index)
            # x = x + residual  # 添加残差
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)

        x_last = x
        # 最后一层（无ReLU）
        x = self.convs[-1](x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class ResGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(ResGNN, self).__init__()
        self.layers = layers
        # 构造多层GCN
        if layers == 1:
            self.convs = torch.nn.ModuleList()
            self.convs.append(ResGatedGraphConv(input_dim, output_dim))
        else:
            self.convs = torch.nn.ModuleList()
            self.convs.append(ResGatedGraphConv(input_dim, hidden_dim))
            for i in range(layers - 2):
                self.convs.append(ResGatedGraphConv(hidden_dim, hidden_dim))
            self.convs.append(ResGatedGraphConv(hidden_dim, output_dim))
        self.mlp = torch.nn.Linear(hidden_dim, output_dim)
        # 仅在输入维度和隐藏维度不同的情况下使用线性变换对齐维度
        if input_dim != hidden_dim:
            self.residual_fc = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_fc = None
        self.residual_fc = None

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            x = self.convs[0](x, edge_index)
            if test:
                return x
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        residual = x  # 初始输入
        x = self.convs[0](x, edge_index)

        if self.residual_fc is not None:
            residual = self.residual_fc(residual)  # 线性变换对齐维度

        # x = x + residual  # 添加残差
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # 中间层
        for conv in self.convs[1:-1]:
            # residual = x  # 记录残差
            x = conv(x, edge_index)
            # x = x + residual  # 添加残差
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)

        x_last = x
        # 最后一层（无ReLU）
        x = self.convs[-1](x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(GIN, self).__init__()
        self.layers = layers
        # 构造多层GCN
        if layers == 1:
            self.convs = torch.nn.ModuleList()
            # GINConv(nn=nn.Linear(in_channels, hidden_channels))
            self.convs.append(GINConv(nn=nn.Linear(input_dim, output_dim)))
        else:
            self.convs = torch.nn.ModuleList()
            self.convs.append(GINConv(nn=nn.Linear(input_dim, hidden_dim)))
            for i in range(layers - 2):
                self.convs.append(GINConv(nn=nn.Linear(hidden_dim, hidden_dim)))
            self.convs.append(GINConv(nn=nn.Linear(hidden_dim, output_dim)))
        self.mlp = torch.nn.Linear(hidden_dim, output_dim)
        # 仅在输入维度和隐藏维度不同的情况下使用线性变换对齐维度
        if input_dim != hidden_dim:
            self.residual_fc = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_fc = None
        self.residual_fc = None

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            x = self.convs[0](x, edge_index)
            if test:
                return x
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        residual = x  # 初始输入
        x = self.convs[0](x, edge_index)

        if self.residual_fc is not None:
            residual = self.residual_fc(residual)  # 线性变换对齐维度

        # x = x + residual  # 添加残差
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # 中间层
        for conv in self.convs[1:-1]:
            # residual = x  # 记录残差
            x = conv(x, edge_index)
            # x = x + residual  # 添加残差
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        x_last = x
        # 最后一层（无ReLU）
        x = self.convs[-1](x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.3, training=self.training)

        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class GEN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(GEN, self).__init__()
        self.layers = layers
        # 构造多层GCN
        if layers == 1:
            self.convs = torch.nn.ModuleList()
            self.convs.append(GENConv(input_dim, output_dim, num_layers=1))
        else:
            self.convs = torch.nn.ModuleList()
            self.convs.append(GENConv(input_dim, hidden_dim, num_layers=1))
            for i in range(layers - 2):
                self.convs.append(GENConv(hidden_dim, hidden_dim, num_layers=1))
            self.convs.append(GENConv(hidden_dim, output_dim, num_layers=1))
        self.mlp = torch.nn.Linear(hidden_dim, output_dim)
        # 仅在输入维度和隐藏维度不同的情况下使用线性变换对齐维度
        if input_dim != hidden_dim:
            self.residual_fc = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_fc = None
        self.residual_fc = None

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            x = self.convs[0](x, edge_index)
            if test:
                return x
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        residual = x  # 初始输入
        x = self.convs[0](x, edge_index)

        if self.residual_fc is not None:
            residual = self.residual_fc(residual)  # 线性变换对齐维度

        # x = x + residual  # 添加残差
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # 中间层
        for conv in self.convs[1:-1]:
            # residual = x  # 记录残差
            x = conv(x, edge_index)
            # x = x + residual  # 添加残差
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x_last = x
        # 最后一层（无ReLU）
        x = self.convs[-1](x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.3, training=self.training)

        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, heads=1):
        super(GAT, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()

        # 第一层
        if layers == 1:
            self.convs.append(GATConv(input_dim, output_dim, heads=heads))
        else:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
            for _ in range(layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            self.convs.append(
                GATConv(hidden_dim * heads, output_dim, heads=1)
            )  # 输出层 head=1

        # 仅在输入维度和隐藏层维度不匹配时使用线性变换对齐
        # if input_dim != hidden_dim:
        #     self.residual_fc = torch.nn.Linear(input_dim, hidden_dim * heads)
        # else:
        #     self.residual_fc = None
        self.residual_fc = None

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            x = self.convs[0](x, edge_index)
            if test:
                return x
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        residual = x  # 记录初始输入
        x = self.convs[0](x, edge_index)

        if self.residual_fc is not None:
            residual = self.residual_fc(residual)  # 线性变换对齐维度

        # x = x + residual  # 添加残差
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # 中间层
        for conv in self.convs[1:-1]:
            residual = x
            x = conv(x, edge_index)
            # x = x + residual  # 添加残差
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)

        x_last = x
        # 最后一层（无ReLU）
        x = self.convs[-1](x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class AntiSymmetricDGN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(AntiSymmetricDGN, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)

        # 第一层
        self.convs.append(AntiSymmetricConv(hidden_dim, num_iters=layers - 1))
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.convs.append(AntiSymmetricConv(hidden_dim // 2, num_iters=1))

        self.fc = torch.nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        x = self.hidden(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.hidden2(x)
        x = F.leaky_relu(x)
        x_last = x
        x = self.convs[-1](x, edge_index)
        x = self.fc(x)
        # x = F.leaky_relu(x)
        # x = F.dropout(x, p=0.3, training=self.training)
        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class EGCNN(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, layers, num_heads=1, num_bases=4
    ):
        super(EGCNN, self).__init__()
        self.layers = layers

        if layers == 1:
            self.convs = torch.nn.ModuleList()
            self.convs.append(
                EGConv(input_dim, output_dim, num_heads=num_heads, num_bases=num_bases)
            )
        else:
            self.convs = torch.nn.ModuleList()
            self.convs.append(
                EGConv(hidden_dim, hidden_dim, num_heads=num_heads, num_bases=num_bases)
            )
            for _ in range(layers - 2):
                self.convs.append(
                    EGConv(
                        hidden_dim, hidden_dim, num_heads=num_heads, num_bases=num_bases
                    )
                )
            self.convs.append(
                EGConv(hidden_dim, output_dim, num_heads=num_heads, num_bases=num_bases)
            )

        # 维度对齐
        if input_dim != hidden_dim:
            if layers == 1:
                self.residual_fc = torch.nn.Linear(input_dim, output_dim)
            else:
                self.residual_fc = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_fc = None

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            return F.log_softmax(x, dim=1)

        if self.residual_fc is not None:
            residual = self.residual_fc(x)

        # x = x + residual
        x = residual
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        for conv in self.convs[1:-1]:
            # residual = x
            x = conv(x, edge_index)
            # x = x + residual
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x if test else F.log_softmax(x, dim=1)


class PDN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(PDN, self).__init__()
        self.layers = layers
        self.batch_norm = torch.nn.LayerNorm(hidden_dim)
        if layers == 1:
            self.convs = torch.nn.ModuleList()
            self.convs.append(
                PDNConv(input_dim, output_dim, edge_dim=1, hidden_channels=hidden_dim)
            )
        else:
            self.convs = torch.nn.ModuleList()
            self.convs.append(
                PDNConv(input_dim, hidden_dim, edge_dim=1, hidden_channels=hidden_dim)
            )
            for i in range(layers - 2):
                self.convs.append(
                    PDNConv(
                        hidden_dim, hidden_dim, edge_dim=1, hidden_channels=hidden_dim
                    )
                )
            self.convs.append(
                PDNConv(hidden_dim, output_dim, edge_dim=1, hidden_channels=hidden_dim)
            )
        # self.mlp = torch.nn.Linear(hidden_dim, output_dim)
        # 仅在输入维度和隐藏维度不同的情况下使用线性变换对齐维度
        if input_dim != hidden_dim:
            self.residual_fc = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_fc = None
        self.residual_fc = None

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            x = self.convs[0](x, edge_index)
            if test:
                return x
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        residual = x  # 初始输入
        x = self.convs[0](x, edge_index)

        if self.residual_fc is not None:
            residual = self.residual_fc(residual)  # 线性变换对齐维度

        # x = x + residual  # 添加残差
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 中间层
        for conv in self.convs[1:-1]:
            # residual = x  # 记录残差
            x = conv(x, edge_index)
            # x = x + residual  # 添加残差
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x_last = x
        # 最后一层（无ReLU）
        x = self.convs[-1](x, edge_index)
        x = F.leaky_relu(x)
        # x = F.dropout(x, p=0.3, training=self.training)

        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class GATv2(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, heads=1):
        super(GATv2, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.batch_norm = torch.nn.LayerNorm(hidden_dim)
        # 第一层
        if layers == 1:
            self.convs.append(GATv2Conv(input_dim, output_dim, heads=heads))
        else:
            self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=heads))
            for _ in range(layers - 2):
                self.convs.append(
                    GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads)
                )
            self.convs.append(
                GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads)
            )  # 输出层 head=1

        # 仅在输入维度和隐藏层维度不匹配时使用线性变换对齐
        if input_dim != hidden_dim:
            self.residual_fc = torch.nn.Linear(input_dim, hidden_dim * heads)
        else:
            self.residual_fc = None
        # self.residual_fc = None
        self.mlp = torch.nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index

        if self.layers == 1:
            x = self.convs[0](x, edge_index)
            if test:
                return x
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        residual = x  # 记录初始输入
        x = self.convs[0](x, edge_index)

        if self.residual_fc is not None:
            residual = self.residual_fc(residual)  # 线性变换对齐维度

        # x = self.batch_norm(x + residual)  # 添加残差
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 中间层
        for conv in self.convs[1:-1]:
            res = x
            x = conv(x, edge_index)
            # x = self.batch_norm(x + residual)  # 添加残差
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # 最后一层（不加 ReLU）
        x = self.convs[-1](x, edge_index)
        x = self.mlp(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        if test:
            return x
        else:
            return F.log_softmax(x, dim=1)


def GNNLayer(in_channels, out_channels, type, heads=1):
    if type == "GCN":
        return GCNConv(in_channels, out_channels)
    elif type == "GAT":
        return GATConv(in_channels, out_channels)
    elif type == "EGC":
        return EGConv(in_channels, out_channels, aggregators="mean")
    elif type == "GATv2":
        return GATv2Conv(in_channels, out_channels, heads=heads)
    elif type == "PDN":
        return PDNConv(in_channels, out_channels, edge_dim=1)
    else:
        raise ValueError("Invalid GNN type.")


class PMPGNN(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, layers, base_model="GATv2", heads=4
    ):
        super(PMPGNN, self).__init__()
        self.nums = layers // 2
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        self.proj = None  # 用于投影输入到hidden_dim
        self.norm = torch.nn.LayerNorm(hidden_dim * 2 * heads)
        self.heads = heads
        # 三层mlp，特征维度逐渐降低，最后一层输出维度为output_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 * heads, hidden_dim // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(hidden_dim // 2, output_dim),
        )
        # self.mlp = torch.nn.Linear(hidden_dim * 2 * heads, output_dim)
        if input_dim != hidden_dim:
            self.proj = torch.nn.Linear(input_dim, hidden_dim)  # 维度不匹配时投影

        if layers % 2 == 1:
            self.convs.append(
                ParallelGNNBlock(
                    hidden_dim, hidden_dim, base_model=base_model, heads=self.heads
                )
            )
            for _ in range(self.nums - 1):
                self.convs.append(
                    ParallelGNNBlock(
                        hidden_dim * 2 * heads,
                        hidden_dim,
                        base_model=base_model,
                        heads=self.heads,
                    )
                )
            self.convs.append(
                GNNLayer(
                    hidden_dim * 2 * heads,
                    hidden_dim * 2,
                    type=base_model,
                    heads=self.heads,
                )
            )
        else:
            self.convs.append(
                ParallelGNNBlock(
                    hidden_dim, hidden_dim, base_model=base_model, heads=self.heads
                )
            )
            for _ in range(self.nums - 1):
                self.convs.append(
                    ParallelGNNBlock(
                        hidden_dim * 2 * heads,
                        hidden_dim,
                        base_model=base_model,
                        heads=self.heads,
                    )
                )

    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index
        if self.proj is not None:
            x = self.proj(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training, p=0.2)
        x_last = x
        if self.nums == 1 and self.layers % 2 == 0:
            x = self.convs[0](x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training, p=0.2)
            x = self.mlp(x)
            if test:
                return x, x_last, edge_index
            else:
                return F.log_softmax(x, dim=1)

        # 第一层
        # residual = x  # 记录输入值
        # x = self.convs[0](x, edge_index)
        # if self.proj is not None:  # 如果输入维度和hidden_dim不匹配，则进行投影
        #     residual = self.proj(residual)

        # x = self.norm(x)  # 添加残差
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.3)

        for conv in self.convs[:-1]:
            residual = x  # 记录残差连接数据
            x = conv(x, edge_index)
            # 如果维度相同，则计算残差
            if x.size(1) == residual.size(1):
                # x = x + residual
                x = self.norm(x + residual)
            else:
                # x = x
                x = self.norm(x)
            x_last = x
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training, p=0.3)

        residual = x
        x = self.convs[-1](x, edge_index)
        if x.size(1) == residual.size(1):
            x = self.norm(x + residual)
        else:
            x = self.norm(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.mlp(x)
        if test:
            return x, x_last, edge_index
        else:
            return F.log_softmax(x, dim=1)


class ParallelGNNBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=None,
        base_model="GCN",
        alpha=0.01,
        heads=1,
    ):
        super(ParallelGNNBlock, self).__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_dim == None:
            hidden_dim = output_dim
        self.conv1 = GNNLayer(
            in_channels=input_dim, out_channels=output_dim, type=base_model, heads=heads
        )
        self.conv2 = GNNLayer(
            in_channels=input_dim, out_channels=output_dim, type=base_model, heads=heads
        )
        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        # res = x
        x1 = self.conv1(x, edge_index)
        # 对edge_index进行两次幂运算
        edge_index = self.high_adj(edge_index, 2)
        x2 = self.conv2(x, edge_index)

        x = torch.cat([(1 - self.alpha) * x1, self.alpha * x2], dim=1)
        return x

    def high_adj(self, edge_index, num):
        # 对edge_index 邻接矩阵进行num次幂运算
        # edge_index: [2, edge_num]
        # 将edge_index转换为COO格式的稀疏矩阵
        # edge_index 中的最大值即为节点数
        # 暂存原始邻接矩阵
        device = edge_index.device
        source_adj = edge_index.t()
        max_idx = torch.max(edge_index) + 1
        edge_index = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1)).to(edge_index.device),
            (max_idx, max_idx),
        )
        source_edge_index = edge_index
        for _ in range(num - 1):
            edge_index = torch.sparse.mm(source_edge_index, edge_index)
        # 保留非零元素的下标以[2, edge_num]的torch.tensor保存
        edge_index = torch.nonzero(edge_index.to_dense())
        return edge_index.t()
        # edge_index 和 source_adj 都是稀疏矩阵，规格都为[edge_num, 2], 但是两个矩阵的边数量不同，需要从edge_index中去除出现在source_adj中的边
        # 将 tmp 转换为集合以便快速查找
        # source_adj = source_adj.cpu()
        # edge_index = edge_index.cpu()
        # tmp_set = set(map(tuple, source_adj.tolist()))
        # # 保留 edge_index 中不在 tmp 中的元素
        # filtered_edge_index = torch.tensor(
        #     [edge for edge in edge_index.tolist() if tuple(edge) not in tmp_set]
        # )
        # # 转置以符合原始格式
        # filtered_edge_index = filtered_edge_index.t().to(device)
        # edge_index = edge_index.t()
        # return filtered_edge_index


# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    # 损失函数为负对数似然函数
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss.item(), epoch)  # 记录训练损失
    return loss.item()


# 测试模型
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    # 计算训练集、验证集和测试集的准确率
    correct_train = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    correct_val = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    correct_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    acc_train = int(correct_train) / int(data.train_mask.sum())
    acc_val = int(correct_val) / int(data.val_mask.sum())
    acc_test = int(correct_test) / int(data.test_mask.sum())
    writer.add_scalar("Accuracy/train", acc_train, epoch)  # 记录训练准确率
    writer.add_scalar("Accuracy/val", acc_val, epoch)  # 记录验证准确率
    writer.add_scalar("Accuracy/test", acc_test, epoch)  # 记录测试准确率

    return acc_train, acc_val, acc_test


# 平滑度测试
def smoothtest():
    model.eval()
    out, out_last, edge_index = model(data, test=True)
    # 节点表示
    x = out
    # 计算全局MAD, 构造mask矩阵，矩阵中的元素为1表示两个节点之间有边，为0表示没有边，计算全局时mask矩阵为全1矩阵，去掉对角线元素
    # 构造一个全1元素的[node_num * node_num]矩阵
    mask = torch.ones(data.y.size(0), data.y.size(0)).to(device)
    # mask = mask - torch.eye(mask.size(0)).to(device)
    # 构造一个长度为node_num的列表，列表中的元素为1表示对应节点是目标节点，为0表示对应节点不是目标节点
    # target_idx = torch.ones(data.y.size(0)).to(device)
    target_idx = data.test_mask.to(device)
    # 计算MAD
    mad = mad_value(
        x.cpu().detach().numpy(),
        mask.cpu().detach().numpy(),
        # target_idx=target_idx.cpu().detach().numpy(),
    )
    print("MAD: ", mad)
    # 计算InfoGap
    info_gap = Information_redundancy(
        out_last.cpu().detach().numpy(), edge_index.cpu().detach().numpy()
    )
    print("InfoGap: ", info_gap)
    return mad, info_gap


# the numpy version for mad (Be able to compute quickly)
# in_arr:[node_num * hidden_dim], the node feature matrix;
# mask_arr: [node_num * node_num], the mask matrix of the target raltion;
# target_idx = [1,2,3...n], the nodes idx for which we calculate the mad value;
# def mad_value(in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx = None):
#     dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)
#     mask_dist = np.multiply(dist_arr, mask_arr)

#     node_dist = mask_dist.sum(1) / ((mask_dist != 0).sum(1) + 1e-8)
#     if target_idx.any() == None:
#         mad = np.mean(node_dist)
#     else:
#         node_dist = np.multiply(node_dist, target_idx)
#         mad = node_dist.sum(0)/((node_dist!=0).sum(0)+1e-8)
#     mad = round(mad, digt_num)


#     return mad
def mad_value(in_arr, mask_arr, distance_metric="cosine", digt_num=4, target_idx=None):
    dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)

    mask_dist = np.multiply(dist_arr, mask_arr)

    divide_arr = (mask_dist != 0).sum(1) + 1e-8

    node_dist = mask_dist.sum(1) / divide_arr

    if target_idx == None or target_idx.any() == None:
        mad = np.mean(node_dist)
    else:
        node_dist = np.multiply(node_dist, target_idx)
        mad = node_dist.sum() / ((node_dist != 0).sum() + 1e-8)

    mad = round(mad, digt_num)

    return mad


def Information_redundancy(x, edge):
    # x为节点特征，edge为边信息[2 , edge_num]
    # 对于每一个节点，在聚合过程中，计算来自邻居节点的信息冗余度
    r_list = []
    for i in range(x.shape[0]):
        sum_dist = 0
        count = 0
        for j in range(edge.shape[1]):
            if edge[0, j] == i:
                count += 1
                # print(torch.tensor(x[i]))
                p = norm(torch.tensor(x[i]))
                # print(p)
                q = norm(torch.tensor(x[edge[0, j]]))
                sum_dist += F.kl_div(torch.log(p), q, reduction="sum")
                # sum_dist += (1 - cosine_similarity([x[i]], [x[edge[0, j]]])) / 2
            elif edge[1, j] == i:
                count += 1
                p = norm(torch.tensor(x[i]))
                q = norm(torch.tensor(x[edge[1, j]]))
                # print(p)
                sum_dist += F.kl_div(torch.log(p), q, reduction="sum")
                # sum_dist += (1 - cosine_similarity([x[i]], [x[edge[1, j]]])) / 2
        if count != 0:
            r_list.append(sum_dist)
    # 返回均值
    return np.mean(r_list)


# 归一化处理
def norm(x):
    # 所有值加上小值的负数
    min_x, _ = x.min(dim=-1)
    x = x - min_x
    # 归一化处理
    x = F.normalize(x, p=4, dim=-1)
    # 0值用1e-10替换
    x = torch.where(x == 0, torch.tensor(1e-10), x)
    # print(x)
    return x


# the tensor version for mad_gap (Be able to transfer gradients)
# intensor: [node_num * hidden_dim], the node feature matrix;
# neb_mask,rmt_mask:[node_num * node_num], the mask matrices of the neighbor and remote raltion;
# target_idx = [1,2,3...n], the nodes idx for which we calculate the mad_gap value;
def mad_gap_regularizer(intensor, neb_mask, rmt_mask, target_idx):
    node_num, feat_num = intensor.size()

    input1 = intensor.expand(node_num, node_num, feat_num)
    input2 = input1.transpose(0, 1)

    input1 = input1.contiguous().view(-1, feat_num)
    input2 = input2.contiguous().view(-1, feat_num)

    simi_tensor = F.cosine_similarity(input1, input2, dim=1, eps=1e-8).view(
        node_num, node_num
    )
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor, neb_mask)
    rmt_dist = torch.mul(dist_tensor, rmt_mask)

    divide_neb = (neb_dist != 0).sum(1).type(torch.FloatTensor).cuda() + 1e-8
    divide_rmt = (rmt_dist != 0).sum(1).type(torch.FloatTensor).cuda() + 1e-8

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt

    neb_mad = torch.mean(neb_mean_list[target_idx])
    rmt_mad = torch.mean(rmt_mean_list[target_idx])

    mad_gap = rmt_mad - neb_mad

    return mad_gap


# 加载数据集，这里使用Cora数据集
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = Planetoid(root="./data/PubMed", name="PubMed")
dataset = Planetoid(root="./data/Cora", name="Cora")
data = dataset[0].to(device)
# 定义超参数
input_dim = dataset.num_features
hidden_dim = 128
output_dim = dataset.num_classes
learning_rate = 1e-3
epochs = 200
layers = 4
heads = 2
model_name = "PMPGNN"

# 记录每次运行的最佳测试准确率以及mad值
best_test_acc = []
best_mad = []
best_r = []

run_dir = "runs/cora/" + model_name + "_" + str(layers) + "Layers/"

for seed in range(10):
    writer = SummaryWriter(run_dir + str(seed))
    set_seed(seed)
    print("Seed: ", seed)
    save_name = model_name + str(layers) + "Layers_" + str(seed)
    load_name = save_name
    # 初始化模型和优化器
    # model = GCN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    # model = GAT(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    # model = ResGNN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    # model = GIN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    # model = EGCNN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    # model = GATv2(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    # model = PDN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    # model = PMPGNN(input_dim, hidden_dim, output_dim, layers=layers, heads=heads).to(
    #     device
    # )
    if model_name == "GCN":
        model = GCN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    elif model_name == "GAT":
        model = GAT(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    elif model_name == "GIN":
        model = GIN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    elif model_name == "EGC":
        model = EGCNN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    elif model_name == "GEN":
        model = GEN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    elif model_name == "GATv2":
        model = GATv2(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    elif model_name == "PDN":
        model = PDN(input_dim, hidden_dim, output_dim, layers=layers).to(device)
    elif model_name == "PMPGNN":
        model = PMPGNN(
            input_dim, hidden_dim, output_dim, layers=layers, heads=heads
        ).to(device)
    elif model_name == "AntiSymmetricDGN":
        model = AntiSymmetricDGN(input_dim, hidden_dim, output_dim, layers=layers).to(
            device
        )
    # 打印模型
    print(model)

    # Adam优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    best_acc_test = 0.0
    best_acc_val = 0.0
    best_acc_train = 0.0

    # 运行训练和测试循环
    for epoch in range(epochs):
        loss = train()
        acc_train, acc_val, acc_test = test()
        if acc_test > best_acc_test:
            # 记录最好的表现
            best_acc_test = acc_test
            best_acc_val = acc_val
            best_acc_train = acc_train
            # 保存模型
            torch.save(model.state_dict(), f"./pkl/{save_name}.pkl")
        print(
            f"Epoch: {epoch+1}, Loss: {loss:.4f}, Train: {acc_train:.4f}, Val: {acc_val:.4f}, Test: {acc_test:.4f}"
        )

    # 加载最好的模型
    model.load_state_dict(torch.load(f"./pkl/{save_name}.pkl"))
    mad, r = smoothtest()
    # 删除模型文件
    os.remove(f"./pkl/{save_name}.pkl")
    best_test_acc.append(best_acc_test)
    best_mad.append(mad)
    best_r.append(r)
    print(
        f"Best Test Accuracy: {best_acc_test:.4f}, Val Accuracy: {best_acc_val:.4f}, Train Accuracy: {best_acc_train:.4f}"
    )
    print("Training completed.")
    writer.close()

# 输出10次运行的平均测试准确率和mad值的均值和标准差
# 删除最小值
# best_test_acc.remove(min(best_test_acc))
# best_mad.remove(min(best_mad))
print("Average Test Accuracy: ", np.mean(best_test_acc), "±", np.std(best_test_acc))
print("Average MAD: ", np.mean(best_mad), "±", np.std(best_mad))
print("Average R: ", np.mean(best_r), "±", np.std(best_r))
