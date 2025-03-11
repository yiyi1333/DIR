import torch

class TestGNN:
    # def CrossGNN_forward(self, edge_index, num):
    #     # 对edge_index 邻接矩阵进行num次幂运算
    #     # edge_index: [2, edge_num] -- torch.Size([2, edge_num])
    #     # 将edge_index转换为COO格式的稀疏矩阵
    #     max_idx = torch.max(edge_index) + 1  # 节点数
    #     edge_index = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (max_idx, max_idx))
    #     source_edge_index = edge_index.t()
    #     for i in range(num):
    #         edge_index = torch.sparse.mm(source_edge_index, edge_index)
    #     # 保留非零元素的下标以[2, edge_num]的torch.tensor保存
    #     edge_index = torch.nonzero(edge_index.to_dense())
    #     # edge_index 现在的格式是[edge_num, 2],需要转换为[2, edge_num]
    #     edge_index = edge_index.t()
    #     return edge_index
    def high_adj(self, edge_index, num):
        # 对edge_index 邻接矩阵进行num次幂运算
        # edge_index: [2, edge_num] -- torch.Size([2, 10556])
        # 将edge_index转换为COO格式的稀疏矩阵
        # edge_index 中的最大值即为节点数
        max_idx = torch.max(edge_index) + 1
        print("max_idx: ", max_idx)
        edge_index = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (max_idx, max_idx))
        source_edge_index = edge_index
        for i in range(num - 1):
            edge_index = torch.sparse.mm(source_edge_index, edge_index)
        # 保留非零元素的下标以[2, edge_num]的torch.tensor保存
        edge_index = torch.nonzero(edge_index.to_dense())
        # edeg_index 现在的格式是[edge_num, 2],需要转换为[2, edge_num]
        edge_index = edge_index.t()
        return edge_index

# 测试函数
def test_CrossGNN_forward():
    # 创建 GNN 测试类
    gnn = TestGNN()

    # 构造一个简单的图
    # 假设图中有 4 个节点，边的连接关系如下：
    # 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0
    edge_index = torch.tensor([[0, 1, 2, 3],  # 源节点
                               [1, 2, 3, 0]])  # 目标节点

    print("原始一阶邻接矩阵的边索引：")
    print(edge_index)

    # 计算二阶邻接矩阵
    num = 2  # 幂次数
    second_order_edge_index = gnn.high_adj(edge_index, num)

    print("\n二阶邻接矩阵的边索引：")
    print(second_order_edge_index)

# 运行测试
test_CrossGNN_forward()
