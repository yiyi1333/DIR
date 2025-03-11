import copy
import torch
import argparse
from torch_geometric.data import DataLoader
from model.PMPGNN import PMPGNN
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    GATConv,
    ResGatedGraphConv,
    AntiSymmetricConv,
    GINConv,
    PDNConv,
    GENConv,
)


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv
from utils.mask import set_masks, clear_masks

import os
import numpy as np
import os.path as osp
from torch.autograd import grad
from utils.logger import Logger
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.get_subgraph import split_graph, relabel
from datasets.graphsst2_dataset import get_dataset, get_dataloader
from gnn import GraphSST2Net
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# def check_gradients(model):
#     for name, param in model.named_parameters():
#         if param.requires_grad and param.grad is not None:
#             print(f"{name} grad mean: {param.grad.mean().item():.6f}, grad std: {param.grad.std().item():.6f}")
#             break  # 只打印一个参数即可验证


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCN, self).__init__()
        # 定义两层 GCN 卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # 最后一层全连接层用于分类
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # 从数据中提取节点特征和边信息
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 第一层 GCN 卷积并应用激活函数 ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 第二层 GCN 卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 使用全局池化将节点特征聚合为图级特征
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        # 全连接层输出分类结果
        x = self.fc(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super(GAT, self).__init__()
        # 定义两层 GAT 卷积层
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)

        # 最后一层全连接层用于分类
        self.fc = torch.nn.Linear(hidden_channels * heads, num_classes)

    def forward(self, data):
        # 从数据中提取节点特征和边信息
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = F.relu(x)

        x = self.gat2(x, edge_index)
        x = F.relu(x)

        # 使用全局池化将节点特征聚合为图级特征
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels * heads]

        # 全连接层输出分类结果
        x = self.fc(x)

        return x


class AntiSymmetricDGN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=1):
        super(AntiSymmetricDGN, self).__init__()
        self.anticonv = AntiSymmetricConv(in_channels, num_iters=6)

        # 最后一层全连接层用于分类
        self.fc = torch.nn.Linear(in_channels, num_classes)

    def forward(self, data):
        # 从数据中提取节点特征和边信息
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.anticonv(x, edge_index)

        # 使用全局池化将节点特征聚合为图级特征
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels * heads]

        # 全连接层输出分类结果
        x = self.fc(x)

        return x


class ResGatedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(ResGatedGCN, self).__init__()
        # 定义两层 ResGatedGraphConv 卷积层
        self.gat1 = ResGatedGraphConv(in_channels, hidden_channels)
        self.gat2 = ResGatedGraphConv(hidden_channels, hidden_channels)

        # 最后一层全连接层用于分类
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # 从数据中提取节点特征和边信息
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = F.relu(x)

        x = self.gat2(x, edge_index)
        x = F.relu(x)

        # 使用全局池化将节点特征聚合为图级特征
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels * heads]

        # 全连接层输出分类结果
        x = self.fc(x)

        return x


class GENNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GENNet, self).__init__()
        self.genconv1 = GENConv(in_channels, hidden_channels)
        self.genconv2 = GENConv(hidden_channels, hidden_channels)

        # 最后一层全连接层用于分类
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # 从数据中提取节点特征和边信息
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.genconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.genconv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        # 使用全局池化将节点特征聚合为图级特征
        x = global_mean_pool(x, batch)

        # 全连接层输出分类结果
        x = self.fc(x)
        return x


class PDNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(PDNNet, self).__init__()
        self.pdnconv1 = PDNConv(
            in_channels, hidden_channels, edge_dim=1, hidden_channels=hidden_channels
        )
        self.pdnconv2 = PDNConv(
            hidden_channels,
            hidden_channels,
            edge_dim=1,
            hidden_channels=hidden_channels,
        )

        # 最后一层全连接层用于分类
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # 从数据中提取节点特征和边信息
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.pdnconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.pdnconv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        # 使用全局池化将节点特征聚合为图级特征
        x = global_mean_pool(x, batch)

        # 全连接层输出分类结果
        x = self.fc(x)
        return x


class GINNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GINNet, self).__init__()
        self.ginconv1 = GINConv(nn=nn.Linear(in_channels, hidden_channels))
        self.ginconv2 = GINConv(nn=nn.Linear(hidden_channels, hidden_channels))

        # 最后一层全连接层用于分类
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # 从数据中提取节点特征和边信息
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.ginconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.ginconv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        # 使用全局池化将节点特征聚合为图级特征
        x = global_mean_pool(x, batch)

        # 全连接层输出分类结果
        x = self.fc(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training for Causal Feature Learning")
    parser.add_argument("--cuda", default=1, type=int, help="cuda device")
    parser.add_argument(
        "--datadir", default="../data/", type=str, help="directory for datasets."
    )
    parser.add_argument("--epoch", default=50, type=int, help="training iterations")
    parser.add_argument("--reg", default=True, type=bool)
    parser.add_argument(
        "--seed", nargs="?", default="[1,2,3,4,5,6,7,8,9,10]", help="random seed"
    )
    parser.add_argument("--channels", default=128, type=int, help="width of network")
    parser.add_argument("--commit", default="", type=str, help="experiment name")
    parser.add_argument(
        "--type", default="none", type=str, choices=["none", "micro", "macro"]
    )
    parser.add_argument("--input_channels", default=768, type=int)
    # hyper
    parser.add_argument("--pretrain", default=0, type=int, help="pretrain epoch")
    parser.add_argument("--alpha", default=10, type=float, help="invariant loss")
    parser.add_argument("--r", default=0.5, type=float, help="causal_ratio")
    # basic
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--layers", default=4, type=int, help="number of layers")
    parser.add_argument("--heads", default=2, type=int, help="number of heads")
    parser.add_argument(
        "--net_lr", default=1e-5, type=float, help="learning rate for the predictor"
    )
    args = parser.parse_args()
    args.seed = eval(args.seed)

    # dataset
    num_classes = 2
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(dataset_dir="../data/", dataset_name="Teen-Graph", task=None)
    dataloader = get_dataloader(dataset, batch_size=args.batch_size, degree_bias=False)
    train_loader = dataloader["train"]
    val_loader = dataloader["eval"]
    test_loader = dataloader["test"]
    n_train_data, n_val_data = len(train_loader.dataset), len(val_loader.dataset)
    n_test_data = float(len(test_loader.dataset))

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = {
        "train_acc": [],
        "train_pre": [],
        "train_rec": [],
        "train_f1": [],
        "val_acc": [],
        "val_pre": [],
        "val_rec": [],
        "val_f1": [],
        "test_acc": [],
        "test_pre": [],
        "test_rec": [],
        "test_f1": [],
    }

    experiment_name = (
        f"tech-graph.{args.type}.{args.reg}.{args.commit}.netlr_{args.net_lr}.batch_{args.batch_size}"
        f".channels_{args.channels}.pretrain_{args.pretrain}.r_{args.r}.alpha_{args.alpha}.seed_{args.seed}.{datetime_now}"
    )
    exp_dir = osp.join("local/", experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + "/_output_.log")
    args_print(args, logger)

    for seed in args.seed:
        logger.info("Seed:{:3d}".format(int(seed)))
        set_seed(seed)
        # models and optimizers
        # g = GraphSST2Net(args.channels, num_classes=num_classes).to(device)
        # att_net = CausalAttNet(args.r).to(device)
        # model = GCN(args.input_channels, args.channels, num_classes).to(device)
        # model = GAT(args.input_channels, args.channels, num_classes).to(device)
        # model = ResGatedGCN(args.input_channels, args.channels, num_classes).to(device)
        # model = GINNet(args.input_channels, args.channels, num_classes).to(device)
        # model = PDNNet(args.input_channels, args.channels, num_classes).to(device)
        # model = AntiSymmetricDGN(args.input_channels, args.channels, num_classes).to(device)
        model = PMPGNN(
            args.input_channels,
            args.channels,
            num_classes,
            layers=args.layers,
            heads=args.heads,
        ).to(device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.net_lr)
        # conf_opt = torch.optim.Adam(g.conf_mlp.parameters(), lr=args.net_lr)
        CELoss = nn.CrossEntropyLoss(reduction="mean")
        # EleCELoss = nn.CrossEntropyLoss(reduction="none")

        def train_mode():
            # model.train();att_net.train()
            model.train()

        def val_mode():
            # g.eval();att_net.eval()
            model.eval()

        def test_metrics(loader, att_net):
            acc = 0
            all_preds = []
            all_labels = []

            for graph in loader:
                graph.to(device)
                out = att_net(graph)
                preds = out.argmax(-1).view(-1)
                labels = graph.y.view(-1)

                acc += torch.sum(preds == labels)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            # 计算accuracy
            # acc = float(acc) / len(loader.dataset)

            # 将所有预测和标签合并
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            # 计算precision, recall, f1
            acc = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average="weighted")
            recall = recall_score(all_labels, all_preds, average="weighted")
            f1 = f1_score(all_labels, all_preds, average="weighted")

            return acc, precision, recall, f1

        logger.info(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")
        cnt, last_val_acc = 0, 0
        for epoch in range(args.epoch):
            alpha_prime = args.alpha * (epoch**1.6)
            all_loss, n_bw, all_env_loss = 0, 0, 0
            all_causal_loss, all_conf_loss, all_var_loss = 0, 0, 0
            train_mode()
            for graph in train_loader:
                n_bw += 1
                graph.to(device)
                N = graph.num_graphs

                out = model(graph)
                labels = graph.y.squeeze(-1).long()

                causal_loss = CELoss(out, labels)
                all_causal_loss += causal_loss

                # 更新模型参数
                model_optimizer.zero_grad()
                causal_loss.backward()
                model_optimizer.step()

            all_causal_loss /= n_bw
            all_loss = all_causal_loss  # + all_env_loss if you add other losses
            val_mode()
            with torch.no_grad():
                train_acc, train_pre, train_rec, train_f1 = test_metrics(
                    train_loader, model
                )
                val_acc, val_pre, val_rec, val_f1 = test_metrics(val_loader, model)
                test_acc, test_pre, test_rec, test_f1 = test_metrics(test_loader, model)

                logger.info(
                    "Epoch [{:3d}/{:d}]"
                    "Loss: {:.3f}  "
                    "Train_ACC:{:.3f} Test_ACC:{:.3f}  Val_ACC:{:.3f}  "
                    "Train_Pre:{:.3f} Test_Pre:{:.3f}  Val_Pre:{:.3f}  "
                    "Train_Recall:{:.3f} Test_Recall:{:.3f}  Val_Recall:{:.3f}  "
                    "Train_F1:{:.3f} Test_F1:{:.3f}  Val_F1:{:.3f}  ".format(
                        epoch,
                        args.epoch,
                        all_loss,
                        train_acc,
                        val_acc,
                        test_acc,
                        train_pre,
                        val_pre,
                        test_pre,
                        train_rec,
                        val_rec,
                        test_rec,
                        train_f1,
                        val_f1,
                        test_f1,
                    )
                )

                if val_acc > last_val_acc:
                    last_val_acc = val_acc
                    last_val_pre = val_pre
                    last_val_rec = val_rec
                    last_val_f1 = val_f1

                    last_train_acc = train_acc
                    last_train_pre = train_pre
                    last_train_rec = train_rec
                    last_train_f1 = train_f1

                    last_test_acc = test_acc
                    last_test_pre = test_pre
                    last_test_rec = test_rec
                    last_test_f1 = test_f1

        # 将更新后的指标保存到 all_info 字典中
        all_info["train_acc"].append(last_train_acc)
        all_info["train_pre"].append(last_train_pre)
        all_info["train_rec"].append(last_train_rec)
        all_info["train_f1"].append(last_train_f1)

        all_info["val_acc"].append(last_val_acc)
        all_info["val_pre"].append(last_val_pre)
        all_info["val_rec"].append(last_val_rec)
        all_info["val_f1"].append(last_val_f1)

        all_info["test_acc"].append(last_test_acc)
        all_info["test_pre"].append(last_test_pre)
        all_info["test_rec"].append(last_test_rec)
        all_info["test_f1"].append(last_test_f1)

        # torch.save(g.cpu(), osp.join(exp_dir, 'predictor-%d.pt' % seed))
        # torch.save(att_net.cpu(), osp.join(exp_dir, 'attention_net-%d.pt' % seed))
        logger.info("=" * 100)
    # 删除all_info["test_acc"]中的最小值
    min_index = all_info["test_acc"].index(min(all_info["test_acc"]))
    all_info["test_acc"].pop(min_index)
    all_info["test_pre"].pop(min_index)
    all_info["test_rec"].pop(min_index)
    all_info["test_f1"].pop(min_index)
    logger.info(
        "Test ACC:{:.4f}±{:.4f}  Train ACC:{:.4f}±{:.4f}  Val ACC:{:.4f}±{:.4f}".format(
            torch.tensor(all_info["test_acc"]).mean(),
            torch.tensor(all_info["test_acc"]).std(),
            torch.tensor(all_info["train_acc"]).mean(),
            torch.tensor(all_info["train_acc"]).std(),
            torch.tensor(all_info["val_acc"]).mean(),
            torch.tensor(all_info["val_acc"]).std(),
        )
    )
    logger.info(
        "Test Pre:{:.4f}±{:.4f}  Train Pre:{:.4f}±{:.4f}  Val Pre:{:.4f}±{:.4f}".format(
            torch.tensor(all_info["test_pre"]).mean(),
            torch.tensor(all_info["test_pre"]).std(),
            torch.tensor(all_info["train_pre"]).mean(),
            torch.tensor(all_info["train_pre"]).std(),
            torch.tensor(all_info["val_pre"]).mean(),
            torch.tensor(all_info["val_pre"]).std(),
        )
    )
    logger.info(
        "Test Recall:{:.4f}±{:.4f}  Train Recall:{:.4f}±{:.4f}  Val Recall:{:.4f}±{:.4f}".format(
            torch.tensor(all_info["test_rec"]).mean(),
            torch.tensor(all_info["test_rec"]).std(),
            torch.tensor(all_info["train_rec"]).mean(),
            torch.tensor(all_info["train_rec"]).std(),
            torch.tensor(all_info["val_rec"]).mean(),
            torch.tensor(all_info["val_rec"]).std(),
        )
    )
    logger.info(
        "Test F1:{:.4f}±{:.4f}  Train F1:{:.4f}±{:.4f}  Val F1:{:.4f}±{:.4f}".format(
            torch.tensor(all_info["test_f1"]).mean(),
            torch.tensor(all_info["test_f1"]).std(),
            torch.tensor(all_info["train_f1"]).mean(),
            torch.tensor(all_info["train_f1"]).std(),
            torch.tensor(all_info["val_f1"]).mean(),
            torch.tensor(all_info["val_f1"]).std(),
        )
    )
