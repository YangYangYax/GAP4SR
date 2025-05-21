# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, constant_
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.nn import GATConv
from torch.nn import Module
import torch.nn.functional as F
import numpy as np

import torch
from torch.nn.functional import one_hot


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 28238,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: Tensor = None):
        """
不同数据集num_classes不一样,
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='mean',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1



class GlobalGNN(Module):
    def __init__(self, args):
        super(GlobalGNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        in_channels = hidden_channels = self.hidden_size
        self.num_layers = len(args.sample_size)
        self.dropout = nn.Dropout(args.gnn_dropout_prob)
        self.gcn = GCNConv(self.hidden_size, self.hidden_size)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))

    def forward(self, x, adjs, attr, asymmetric_mode=False):
        """
        GlobalGNN 的前向传播，支持非对称增强
        :param x: 输入特征
        :param adjs: 采样的子图邻接关系
        :param attr: 边的属性（权重）
        :param asymmetric_mode: 是否开启非对称增强
        """
        xs = []
        x_all = x

        if self.num_layers > 1:
            for i, (edge_index, e_id, size) in enumerate(adjs):
                weight = attr[e_id].view(-1).type(torch.float)

                # **非对称增强2：扰动边权重**
                if asymmetric_mode:
                    weight = weight * (1.0 + 0.08 * torch.randn_like(weight)) # 让边权重在 90%~110% 之间变化

                x = x_all
                if len(list(x.shape)) < 2:
                    x = x.unsqueeze(0)
                x = self.gcn(x, edge_index, weight)

                # **SAGE**
                x_target = x[:size[1]]  # 目标节点总是排在前面
                x = self.convs[i]((x, x_target), edge_index)

                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = self.dropout(x)

        else:
            # **只有 1-hop 的情况**
            edge_index, e_id, size = adjs.edge_index, adjs.e_id, adjs.size
            x = x_all
            x = self.dropout(x)
            weight = attr[e_id].view(-1).type(torch.float)

            # **非对称增强3：扰动边权重**
            if asymmetric_mode:
                weight = weight * (0.8 + 0.4 * torch.rand_like(weight))  # 让边权重在 80%~120% 之间变化

            if len(list(x.shape)) < 2:
                x = x.unsqueeze(0)
            x = self.gcn(x, edge_index, weight)
            x_target = x[:size[1]]
            x = self.convs[-1]((x, x_target), edge_index)
        xs.append(x)
        return torch.cat(xs, 0)



class InceptionDWConv2d(nn.Module):
    """Inception depthwise convolution（Inception深度卷积）"""

    def __init__(self, in_channels=256, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        # 计算每个卷积分支的通道数
        gc = int(in_channels * branch_ratio)

        # 定义深度卷积层（深度可分离卷积的空间卷积部分）
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        # 定义深度卷积层（深度可分离卷积的宽度方向卷积部分）
        self.dwconv_w = nn.Conv2d(
            gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc
        )

        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class DNorm(nn.Module):
    def __init__(
            self,
            dim1=0, dim2=2
    ):
        super().__init__()
        self.dim1 = dim1  # softmax操作的维度
        self.dim2 = dim2  # 标准化操作的维度
        self.softmax = nn.Softmax(dim=self.dim1)  # 初始化softmax层

    def forward(self, attn: Tensor) -> Tensor:
        attn = self.softmax(attn)  # bs,n,S: 对每个元素进行softmax归一化
        attn = attn / torch.sum(attn, dim=self.dim2, keepdim=True)  # bs,n,S: 对每个元素进行标准化，确保每行和为1
        return attn


class ContraNorm(nn.Module):
    def __init__(self, dim=64, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False,
                 identity=False):
        super().__init__()
        if learnable and scale > 0:
            if positive:
                scale_init = math.log(scale)
        else:
            scale_init = scale
        self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1, 2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1 + self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x


class GCL4SR(nn.Module):
    def __init__(self, args, global_graph):
        super(GCL4SR, self).__init__()
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id) if self.cuda_condition else "cpu")
        self.global_graph = global_graph.to(self.device)
        self.global_gnn = GlobalGNN(args)

        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # sequence encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size,
                                                        nhead=args.num_attention_heads,
                                                        dim_feedforward=4 * args.hidden_size,
                                                        dropout=args.attention_probs_dropout_prob,
                                                        activation=args.hidden_act)
        self.item_encoder_old = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_hidden_layers)

        self.item_encoder = InceptionDWConv2d()
        # self.item_encoder1 = InceptionDWConv2d(in_channels=64)
        self.item_encoder1 = InceptionDWConv2d(in_channels=64)
        self.item_encoder2 = InceptionDWConv2d(in_channels=983)
        self.num_layers = args.num_hidden_layers

        self.Contra_Norm = ContraNorm()
        # AttNet
        self.w_1 = nn.Parameter(torch.Tensor(2 * args.hidden_size, args.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(args.hidden_size, 1))
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # fast run with mmd
        self.w_g = nn.Linear(args.hidden_size, 1)
        self.w_e = nn.Linear(args.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.linear_transform = nn.Linear(3 * args.hidden_size, args.hidden_size, bias=False)
        self.gnndrop = nn.Dropout(args.gnn_dropout_prob)

        self.criterion = Poly1CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.args = args
        self.apply(self._init_weights)

        # user-specific gating
        self.gate_item = Variable(torch.zeros(args.hidden_size, 1).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_user = Variable(torch.zeros(args.hidden_size, args.max_seq_length).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_item = torch.nn.init.xavier_uniform_(self.gate_item)
        self.gate_user = torch.nn.init.xavier_uniform_(self.gate_user)

        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.fc1 = nn.Linear(args.hidden_size, 64)
        self.fc2 = nn.Linear(64, args.hidden_size)

    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def MMD_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        if self.args.fast_run:
            source = source.view(-1, self.args.max_seq_length)
            target = target.view(-1, self.args.max_seq_length)
            batch_size = int(source.size()[0])
            loss_all = []
            kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                           fix_sigma=fix_sigma)
            xx = kernels[:batch_size, :batch_size]
            yy = kernels[batch_size:, batch_size:]
            xy = kernels[:batch_size, batch_size:]
            yx = kernels[batch_size:, :batch_size]
            loss = torch.mean(xx + yy - xy - yx)
            loss_all.append(loss)
        else:
            source = source.view(-1, self.args.max_seq_length, self.args.hidden_size)
            target = target.view(-1, self.args.max_seq_length, self.args.hidden_size)
            batch_size = int(source.size()[1])
            loss_all = []
            for i in range(int(source.size()[0])):
                kernels = self.guassian_kernel(source[i], target[i], kernel_mul=kernel_mul, kernel_num=kernel_num,
                                               fix_sigma=fix_sigma)
                xx = kernels[:batch_size, :batch_size]
                yy = kernels[batch_size:, batch_size:]
                xy = kernels[:batch_size, batch_size:]
                yx = kernels[batch_size:, :batch_size]
                loss = torch.mean(xx + yy - xy - yx)
                loss_all.append(loss)
        return sum(loss_all) / len(loss_all)

    def GCL_loss(self, hidden, hidden_norm=True, temperature=1.0):
        """
        计算 GCL（Global Contrastive Learning）损失，同时结合：
        - 正样本 InfoNCE 损失
        - 负样本学习（并赋予可学习的权重 gamma）
        - KL 散度（用于对齐两个视图的分布）

        使得模型可以动态调整不同损失的贡献，增强对比学习效果。
        """
        batch_size = hidden.shape[0] // 2
        LARGE_NUM = 1e9

        if not hasattr(self, "temperature"):
            self.temperature = nn.Parameter(torch.tensor(temperature))
        if not hasattr(self, "alpha"):
            self.alpha = nn.Parameter(torch.tensor(1.0))  # InfoNCE 权重
        if not hasattr(self, "beta"):
            self.beta = nn.Parameter(torch.tensor(0.1))  # KL 散度权重
        if not hasattr(self, "gamma"):
            self.gamma = nn.Parameter(torch.tensor(0.5))  # 负样本学习权重

        if hidden_norm:
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)

        hidden1, hidden2 = torch.split(hidden, batch_size, dim=0)

        # ** 计算标准 GCL**
        labels = torch.arange(batch_size, device=hidden.device)
        masks = torch.nn.functional.one_hot(labels, batch_size)

        logits_aa = torch.matmul(hidden1, hidden1.T) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2.T) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2.T) / self.temperature
        logits_ba = torch.matmul(hidden2, hidden1.T) / self.temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)

        gcl_loss = (loss_a + loss_b)  # **标准 GCL Loss**

        # **计算 KL 散度**
        def kl_divergence_loss(z1, z2):
            p = F.log_softmax(z1, dim=-1)
            q = F.softmax(z2, dim=-1)
            return F.kl_div(p, q, reduction="batchmean")

        loss_kl = kl_divergence_loss(hidden1, hidden2)

        # **7. 计算负样本损失**
        f = lambda x: torch.exp(x / self.temperature)

        # **计算负样本相似度**
        neg_sim1 = f(torch.mm(hidden1, hidden1.T))  # 负样本相似度（视图1）
        neg_sim2 = f(torch.mm(hidden2, hidden2.T))  # 负样本相似度（视图2）

        # **选择最难区分的负样本**
        hard_neg1 = torch.topk(neg_sim1, k=5, dim=1)[0].mean(1)  # 取前 5 个最难负样本
        hard_neg2 = torch.topk(neg_sim2, k=5, dim=1)[0].mean(1)

        loss_neg = torch.log(hard_neg1 + hard_neg2 + 1e-8).mean()  # 负样本损失

        # ** 计算最终损失**
        final_loss = gcl_loss + self.alpha * loss_a.mean() + self.beta * loss_kl + self.gamma * loss_neg

        return final_loss


    def gnn_encode(self, items, asymmetric_mode=False):
        """
        通过子图采样生成全局 GNN 视图
        """
        subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.sample_size,
                                           shuffle=False, num_workers=0, batch_size=items.shape[0])

        g_adjs = []
        s_nodes = []
        for (b_size, node_idx, adjs) in subgraph_loaders:
            if type(adjs) == list:
                g_adjs = [adj.to(items.device) for adj in adjs]
            else:
                g_adjs = adjs.to(items.device)

            n_idxs = node_idx.to(items.device)
            s_nodes = self.item_embeddings(n_idxs).squeeze()

        attr = self.global_graph.edge_attr.to(items.device)

        # **调用 GlobalGNN 并传递 asymmetric_mode**
        g_hidden = self.global_gnn(s_nodes, g_adjs, attr, asymmetric_mode=asymmetric_mode)

        return g_hidden





    def final_att_net(self, seq_mask, hidden):
        batch_size = hidden.shape[0]
        lens = hidden.shape[1]
        pos_emb = self.position_embeddings.weight[:lens]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        seq_hidden = torch.sum(hidden * seq_mask, -2) / torch.sum(seq_mask, 1)
        seq_hidden = seq_hidden.unsqueeze(-2).repeat(1, lens, 1)
        item_hidden = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        item_hidden = torch.tanh(item_hidden)
        score = torch.sigmoid(self.linear_1(item_hidden) + self.linear_2(seq_hidden))
        att_score = torch.matmul(score, self.w_2)
        att_score_masked = att_score * seq_mask
        output = torch.sum(att_score_masked * hidden, 1)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -10000.0).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data):

        user_ids = data[0]
        inputs = data[1]

        seq = inputs.flatten()
        seq_mask = (inputs == 0).float().unsqueeze(-1)
        seq_mask = 1.0 - seq_mask

        # seq_hidden_global_a = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)
        # seq_hidden_global_b = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)
        seq_hidden_global_a = self.gnn_encode(seq, asymmetric_mode=True).view(-1, self.args.max_seq_length,
                                                                              self.args.hidden_size)
        seq_hidden_global_b = self.gnn_encode(seq, asymmetric_mode=False).view(-1, self.args.max_seq_length,
                                                                               self.args.hidden_size)

        #
        key_padding_mask = (inputs == 0)
        attn_mask = self.generate_square_subsequent_mask(self.args.max_seq_length).to(inputs.device)
        seq_hidden_local = self.item_embeddings(inputs)
        seq_hidden_local = self.LayerNorm(seq_hidden_local)
        seq_hidden_local = self.dropout(seq_hidden_local)

        seq_hidden_permute = seq_hidden_local.permute(1, 0, 2)

        encoded_layers = self.item_encoder_old(seq_hidden_permute,
                                               mask=attn_mask,
                                               src_key_padding_mask=key_padding_mask)

        seq_hidden_local = encoded_layers.permute(1, 0, 2)


        target_shape1 = torch.Size([1024, 50, 64])
        if seq_hidden_local.shape == target_shape1:
            seq_hidden_permute = seq_hidden_local.reshape(4, 256, 50, 64)

            # sequence_output = self.item_encoder(seq_hidden_permute)
            sequence_output = self.item_encoder(seq_hidden_permute)
            # sequence_output = encoded_layers.permute(1, 0, 2)

            sequence_output = sequence_output.reshape(1024, 50, 64)
        else:
            seq_hidden_permute = seq_hidden_local.reshape(1, 64, 50, -1)
            sequence_output = self.item_encoder1(seq_hidden_permute)
            sequence_output = sequence_output.reshape(-1, 50, 64)

        # target_shape2 = torch.Size([983, 50, 64])

        seq_hidden_global_a = 0.2 * self.Contra_Norm(seq_hidden_global_a) + seq_hidden_global_a
        seq_hidden_global_b = 0.2 * self.Contra_Norm(seq_hidden_global_b) + seq_hidden_global_b
        # print(seq_hidden_global_a.shape)

        user_emb = self.user_embeddings(user_ids).view(-1, self.args.hidden_size)

        gating_score_a = torch.sigmoid(torch.matmul(seq_hidden_global_a, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)
        gating_score_b = torch.sigmoid(torch.matmul(seq_hidden_global_b, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_b = seq_hidden_global_b * gating_score_b.unsqueeze(2)

        user_seq_a = self.gnndrop(user_seq_a)
        user_seq_b = self.gnndrop(user_seq_b)

        hidden = torch.cat([sequence_output, user_seq_a, user_seq_b], -1)
        hidden = self.linear_transform(hidden)

        return sequence_output, hidden, user_seq_a, user_seq_b, (seq_hidden_global_a, seq_hidden_global_b), seq_mask

    def train_stage(self, data):
        targets = data[2]
        sequence_output, hidden, user_seq_a, user_seq_b, (seq_gnn_a, seq_gnn_b), seq_mask = self.forward(data)
        seq_out = self.final_att_net(seq_mask, hidden)
        seq_out = self.dropout(seq_out)
        test_item_emb = self.item_embeddings.weight[:self.args.item_size]
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, targets)

        sum_a = torch.sum(seq_gnn_a * seq_mask, 1) / torch.sum(seq_mask.float(), 1)
        sum_b = torch.sum(seq_gnn_b * seq_mask, 1) / torch.sum(seq_mask.float(), 1)

        info_hidden = torch.cat([sum_a, sum_b], 0)


        gcl_loss = self.GCL_loss(info_hidden, hidden_norm=True)
        # gcl_loss = self.GCL_loss(info_hidden, hidden_norm=True, temperature=0.5)

        # [B, L, d] to [B, L]️, can reduce training time and memory
        if self.args.fast_run:
            seq_hidden_local = self.w_e(self.item_embeddings(data[1])).squeeze().unsqueeze(0)
            user_seq_a = self.w_g(user_seq_a).squeeze()
            user_seq_b = self.w_g(user_seq_b).squeeze()
        else:
            seq_hidden_local = self.item_embeddings(data[1])
        mmd_loss = self.MMD_loss(seq_hidden_local, user_seq_a) + self.MMD_loss(seq_hidden_local, user_seq_b)

        joint_loss = main_loss + self.args.lam1 * gcl_loss + self.args.lam2 * mmd_loss

        return joint_loss, main_loss, gcl_loss, mmd_loss

    def eval_stage(self, data):
        _, hidden, _, _, _, seq_mask = self.forward(data)
        hidden = self.final_att_net(seq_mask, hidden)

        return hidden
