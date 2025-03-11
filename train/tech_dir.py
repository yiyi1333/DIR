import copy
import torch
import argparse
from torch_geometric.data import DataLoader
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv, GATConv
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


class CausalAttNet(nn.Module):

    def __init__(self, causal_ratio):
        super(CausalAttNet, self).__init__()
        self.conv1 = ARMAConv(in_channels=1029, out_channels=args.channels)
        self.conv2 = ARMAConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 1),
        )
        self.ratio = causal_ratio

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), (
            conf_edge_index,
            conf_edge_attr,
            conf_edge_weight,
        ) = split_graph(data, edge_score, self.ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(
            x, causal_edge_index, data.batch
        )
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (
            (
                causal_x,
                causal_edge_index,
                causal_edge_attr,
                causal_edge_weight,
                causal_batch,
            ),
            (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),
            edge_score,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training for Causal Feature Learning")
    parser.add_argument("--cuda", default=1, type=int, help="cuda device")
    parser.add_argument(
        "--datadir", default="data/", type=str, help="directory for datasets."
    )
    parser.add_argument("--epoch", default=100, type=int, help="training iterations")
    parser.add_argument("--reg", default=True, type=bool)
    parser.add_argument(
        "--seed", nargs="?", default="[1,2,3,4,5,6,7,8,9,10]", help="random seed"
    )
    parser.add_argument("--channels", default=256, type=int, help="width of network")
    parser.add_argument("--commit", default="", type=str, help="experiment name")
    parser.add_argument(
        "--type", default="none", type=str, choices=["none", "micro", "macro"]
    )
    # hyper
    parser.add_argument("--pretrain", default=0, type=int, help="pretrain epoch")
    parser.add_argument("--alpha", default=10, type=float, help="invariant loss")
    parser.add_argument("--r", default=0.50, type=float, help="causal_ratio")
    # basic
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "--net_lr", default=1e-3, type=float, help="learning rate for the predictor"
    )
    parser.add_argument(
        "--dataset", type=str, default="politifact", choices=["politifact", "gossipcop"]
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="bert",
        choices=["profile", "spacy", "bert", "content"],
    )
    args = parser.parse_args()
    args.seed = eval(args.seed)

    # dataset
    num_classes = 2
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(dataset_dir="data/", dataset_name="Tech-Graph", task=None)
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
        g = GraphSST2Net(args.channels, num_classes=num_classes).to(device)
        att_net = CausalAttNet(args.r).to(device)
        model_optimizer = torch.optim.Adam(
            list(g.parameters()) + list(att_net.parameters()), lr=args.net_lr
        )
        conf_opt = torch.optim.Adam(g.conf_mlp.parameters(), lr=args.net_lr)
        CELoss = nn.CrossEntropyLoss(reduction="mean")
        EleCELoss = nn.CrossEntropyLoss(reduction="none")

        def train_mode():
            g.train()
            att_net.train()

        def val_mode():
            g.eval()
            att_net.eval()

        def test_metrics(loader, att_net, predictor):
            acc = 0
            all_preds = []
            all_labels = []

            for graph in loader:
                graph.to(device)

                (
                    (
                        causal_x,
                        causal_edge_index,
                        causal_edge_attr,
                        causal_edge_weight,
                        causal_batch,
                    ),
                    _,
                    _,
                ) = att_net(graph)
                set_masks(causal_edge_weight, g)
                out = predictor(
                    x=causal_x,
                    edge_index=causal_edge_index,
                    edge_attr=causal_edge_attr,
                    batch=causal_batch,
                )
                clear_masks(g)

                preds = out.argmax(-1).view(-1)
                labels = graph.y.view(-1)

                acc += torch.sum(preds == labels)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            # 计算accuracy
            acc = float(acc) / len(loader.dataset)

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

            causal_edge_weights = torch.tensor([]).to(device)
            conf_edge_weights = torch.tensor([]).to(device)
            alpha_prime = args.alpha * (epoch**1.6)
            all_loss, n_bw, all_env_loss = 0, 0, 0
            all_causal_loss, all_conf_loss, all_var_loss = 0, 0, 0
            train_mode()
            for graph in train_loader:
                n_bw += 1
                graph.to(device)
                N = graph.num_graphs
                (
                    (
                        causal_x,
                        causal_edge_index,
                        causal_edge_attr,
                        causal_edge_weight,
                        causal_batch,
                    ),
                    (
                        conf_x,
                        conf_edge_index,
                        conf_edge_attr,
                        conf_edge_weight,
                        conf_batch,
                    ),
                    edge_score,
                ) = att_net(graph)

                set_masks(causal_edge_weight, g)
                causal_rep = g.get_graph_rep(
                    x=causal_x,
                    edge_index=causal_edge_index,
                    edge_attr=causal_edge_attr,
                    batch=causal_batch,
                )
                causal_out = g.get_causal_pred(causal_rep)
                clear_masks(g)
                set_masks(conf_edge_weight, g)
                conf_rep = g.get_graph_rep(
                    x=conf_x,
                    edge_index=conf_edge_index,
                    edge_attr=conf_edge_attr,
                    batch=conf_batch,
                ).detach()
                conf_out = g.get_conf_pred(conf_rep)
                clear_masks(g)
                labels = graph.y.squeeze(-1).long()
                causal_loss = CELoss(causal_out, labels)
                # F.cross_entropy(causal_out, labels, reduction='mean')
                conf_loss = CELoss(conf_out, labels)
                # F.cross_entropy(conf_out, labels, reduction='mean')

                env_loss = 0
                if args.reg:
                    env_loss = torch.tensor([]).to(device)
                    for idx, causal in enumerate(causal_rep):
                        rep_out = g.get_comb_pred(causal, conf_rep)
                        # tmp = EleCELoss(rep_out, graph.y[idx].repeat(rep_out.size(0)).long())
                        # print(graph.y[idx].repeat(rep_out.size(0)))
                        # print(rep_out)
                        tmp = EleCELoss(
                            rep_out, graph.y[idx].repeat(rep_out.size(0)).long()
                        )
                        # F.cross_entropy(rep_out, graph.y[idx].repeat(rep_out.size(0)).long(), reduction='none')
                        causal_loss += alpha_prime * tmp.mean() / causal_rep.size(0)
                        env_loss = torch.cat([env_loss, torch.var(tmp).unsqueeze(0)])
                    env_loss = alpha_prime * env_loss.mean()

                # logger
                all_conf_loss += conf_loss
                all_causal_loss += causal_loss
                all_env_loss += env_loss
                # causal_edge_weights = torch.cat([causal_edge_weights, causal_edge_weight])
                # conf_edge_weights = torch.cat([conf_edge_weights, conf_edge_weight])

                conf_opt.zero_grad()
                conf_loss.backward()
                conf_opt.step()

                model_optimizer.zero_grad()
                (causal_loss + env_loss).backward()
                model_optimizer.step()

            all_env_loss /= n_bw
            all_causal_loss /= n_bw
            all_loss = all_causal_loss + all_env_loss
            val_mode()
            with torch.no_grad():

                train_acc, train_pre, train_rec, train_f1 = test_metrics(
                    train_loader, att_net, g
                )
                val_acc, val_pre, val_rec, val_f1 = test_metrics(val_loader, att_net, g)
                test_acc, test_pre, test_rec, test_f1 = test_metrics(
                    test_loader, att_net, g
                )

                logger.info(
                    "Epoch [{:3d}/{:d}]"
                    "Train_ACC:{:.3f} Test_ACC:{:.3f}  Val_ACC:{:.3f}  "
                    "Train_Pre:{:.3f} Test_Pre:{:.3f}  Val_Pre:{:.3f}  "
                    "Train_Recall:{:.3f} Test_Recall:{:.3f}  Val_Recall:{:.3f}  "
                    "Train_F1:{:.3f} Test_F1:{:.3f}  Val_F1:{:.3f}  ".format(
                        epoch,
                        args.epoch,
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

        torch.save(g.cpu(), osp.join(exp_dir, "predictor-%d.pt" % seed))
        torch.save(att_net.cpu(), osp.join(exp_dir, "attention_net-%d.pt" % seed))
        logger.info("=" * 100)

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
