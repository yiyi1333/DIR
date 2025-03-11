import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
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
            # x = self.mlp(x)
            return x, x_last, edge_index
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
        self, input_dim, hidden_dim, output_dim, layers, base_model="GATv2", heads=1
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
            torch.nn.ReLU(),
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
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=0.2)
        x_last = x
        if self.nums == 1 and self.layers % 2 == 0:
            x = self.convs[0](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=0.2)
            # x = self.mlp(x)
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
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=0.3)

        residual = x
        x = self.convs[-1](x, edge_index)
        if x.size(1) == residual.size(1):
            x = self.norm(x + residual)
        else:
            x = self.norm(x)
        x = F.relu(x)
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
        base_model="GATv2",
        alpha=0.1,
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


x = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
)
# a = np.array(
#     [
#         [1, 1, 1, 1, 0, 0, 0],
#         [1, 1, 0, 0, 1, 0, 0],
#         [1, 0, 1, 1, 0, 0, 1],
#         [1, 0, 1, 1, 0, 1, 1],
#         [0, 1, 0, 0, 1, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0],
#         [0, 0, 1, 1, 0, 0, 1],
#     ]
# )

edge_index = np.array(
    [
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6],
        [0, 1, 2, 3, 0, 1, 4, 0, 2, 3, 6, 0, 2, 3, 5, 6, 1, 4, 3, 5, 2, 3, 6],
    ]
)
# 将x和edge_index转换成torch.tensor
x = torch.tensor(x, dtype=torch.float32)
edge_index = torch.tensor(edge_index, dtype=torch.int64)
device = "cuda:1"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
                sum_dist += (1 - cosine_similarity([x[i]], [x[edge[1, j]]])) / 2
            elif edge[1, j] == i:
                count += 1
                sum_dist += (1 - cosine_similarity([x[i]], [x[edge[1, j]]])) / 2
        if count != 0:
            r_list.append(sum_dist / count)
    # 返回均值
    return 1 - np.mean(r_list)


# def trad(x, a):
#     # 创建一个与x相同规模的矩阵,数据用float64类型
#     y = np.zeros((10, 10))
#     r = np.zeros(10)
#     for i in range(x.shape[0]):
#         sum_dist = 0
#         count = 0
#         for j in range(x.shape[1]):
#             if a[i, j] == 1:
#                 count += 1
#                 y[i] += x[j]
#                 # 计算x[i]与x[j]的余弦距离
#                 sum_dist += cosine_distances([x[i]], [x[j]])
#         if count != 0:
#             y[i] /= count
#             r[i] = sum_dist / count
#     return y, r


# def parallel(x, a):
#     y = np.zeros((10, 10))
#     r = np.zeros(10)
#     # 计算邻接矩阵a的2阶矩阵
#     a2 = np.dot(a, a)
#     a2 = np.minimum(a2, 1)
#     a2 = a2 - a
#     a2 = np.maximum(a2, 0)


#     for i in range(x.shape[0]):
#         count = 0.0
#         for j in range(x.shape[1]):
#             if a[i, j] == 1:
#                 y[i] += x[j] * 0.9
#                 r[i] += cosine_distances([x[i]], [x[j]]) * 0.9
#                 count += 0.9
#             elif a2[i, j] == 1:
#                 y[i] += x[j] * 0.1
#                 r[i] += cosine_distances([x[i]], [x[j]]) * 0.1
#                 count += 0.1
#         y[i] = y[i] / count
#         r[i] = r[i] / count
#     return y, r
def smoothtest(model, data):
    model.eval()
    out, out_last, edge_index = model(data, test=True)
    # 节点表示
    x = out
    # 计算全局MAD, 构造mask矩阵，矩阵中的元素为1表示两个节点之间有边，为0表示没有边，计算全局时mask矩阵为全1矩阵，去掉对角线元素
    # 构造一个全1元素的[node_num * node_num]矩阵
    mask = torch.ones(7, 7).to(device)
    # mask = mask - torch.eye(mask.size(0)).to(device)
    # 构造一个长度为node_num的列表，列表中的元素为1表示对应节点是目标节点，为0表示对应节点不是目标节点
    # target_idx = torch.ones(data.y.size(0)).to(device)
    # target_idx = data.test_mask.to(device)
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
    return mad, info_gap, out


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


if __name__ == "__main__":
    mad_list = []
    info_gap_list = []
    for i in range(100):
        set_seed(i)
        model = PMPGNN(input_dim=7, hidden_dim=7, output_dim=7, layers=2).to(device)
        # model = GATv2(input_dim=7, hidden_dim=7, output_dim=7, layers=2).to(device)
        data = dict()
        # 创建 Data 对象并封装数据
        data = Data(x=x, edge_index=edge_index.contiguous()).to(device)
        # data.edge_index = edge_index
        mad, info_gap, out = smoothtest(model, data)
        mad_list.append(mad)
        info_gap_list.append(info_gap)
        # if mad >= np.max(mad_list):

        # 保留两位小数,打印矩阵out
        print(np.round(out.cpu().detach().numpy(), 2))
        # break
    print("Mean MAD: ", np.mean(mad_list))
    print("Mean InfoGap: ", np.mean(info_gap_list))
