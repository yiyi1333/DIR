import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    EGConv,
    GATv2Conv,
    PDNConv,
    global_mean_pool,
)


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
        self, input_dim, hidden_dim, output_dim, layers, base_model="GATv2", heads=2
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
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim * 2 * heads, output_dim),
        # )

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
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x, edge_index = data.x, data.edge_index
        if self.proj is not None:
            x = self.proj(x)

        for conv in self.convs:
            residual = x  # 记录残差连接数据
            x = conv(x, edge_index)
            # 如果维度相同，则计算残差
            if x.size(1) == residual.size(1):
                x = self.norm(x + residual)
            else:
                x = self.norm(x)
            # x = F.relu(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training, p=0.3)

        # 全局池化操作，将节点特征聚合为图特征
        x = global_mean_pool(x, batch)

        x = self.mlp(x)
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
