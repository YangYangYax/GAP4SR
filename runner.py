
# -*- coding: utf-8 -*-
import numpy as np
import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import sys
import os
import pickle
import re  # 用于匹配已有文件编号

from datasets.build_witg import build_WITG_from_trainset
from dataset import GCL4SRData
from trainer import GCL4SR_Train
from model import GCL4SR
from utils import check_path, set_seed, EarlyStopping, get_matrix_and_num

sys.path.append(os.path.dirname(sys.path[0]))


def main():
    parser = argparse.ArgumentParser()

    # 模型、数据和训练相关参数
    parser.add_argument("--model_name", default='GCL4SR', type=str)
    parser.add_argument("--data_name", default='poetry', type=str)
    parser.add_argument("--data_dir", default='./datasets/home/', type=str)
    parser.add_argument("--output_dir", default='output/', type=str)
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--lr_dc", type=float, default=0.7, help='learning rate decay.')
    parser.add_argument("--lr_dc_step", type=int, default=5, help='learning rate decay step')
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # Transformer 相关参数
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--hidden_act", default="gelu", type=str, help="activation function")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--max_seq_length", default=50, type=int, help="max sequence length")
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of model")
    parser.add_argument("--seed", default=2020, type=int, help="random seed")
    parser.add_argument("--log_freq", type=int, default=1, help="log frequency per epoch")
    parser.add_argument("--patience", default=10, type=int, help="early stopping patience")

    # 图神经网络参数
    parser.add_argument("--gnn_dropout_prob", type=float, default=0.5, help="gnn dropout")
    parser.add_argument("--use_renorm", type=bool, default=True, help="use re-normalization when building witg")
    parser.add_argument("--use_scale", type=bool, default=False, help="use scale when building witg")
    parser.add_argument("--fast_run", type=bool, default=True, help="enable fast mode to reduce training time/memory")
    parser.add_argument("--sample_size", default=[20, 20], type=list, help="gnn sample size")
    parser.add_argument("--lam1", type=float, default=1, help="loss lambda 1")
    parser.add_argument("--lam2", type=float, default=1, help="loss lambda 2")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = os.path.join(args.data_dir, args.data_name + '.txt')
    user_num, item_num, valid_rating_matrix, test_rating_matrix = get_matrix_and_num(args.data_file)

    train_data = pickle.load(open(os.path.join(args.data_dir, 'train.pkl'), 'rb'))
    valid_data = pickle.load(open(os.path.join(args.data_dir, 'valid.pkl'), 'rb'))
    test_data = pickle.load(open(os.path.join(args.data_dir, 'test.pkl'), 'rb'))

    args.item_size = item_num
    args.user_size = user_num

    # 生成文件夹前缀
    args_str = f"{args.model_name}-{args.data_name}-{args.sample_size}"

    # 查找已有的文件夹编号
    existing_dirs = [
        d for d in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, d)) and d.startswith(args_str)
    ]

    numbers = []
    pattern = re.compile(rf"{re.escape(args_str)}_(\d+)")  # 正则匹配编号

    for d in existing_dirs:
        match = pattern.search(d)
        if match:
            numbers.append(int(match.group(1)))

    # 计算新的编号
    next_number = max(numbers) + 1 if numbers else 1
    checkpoint_dir = os.path.join(args.output_dir, f"{args_str}_{next_number}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 记录日志文件
    args.log_file = os.path.join(checkpoint_dir, args_str + '.txt')
    with open(args.log_file, 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(str(args) + '\n')

    args.train_matrix = valid_rating_matrix
    args.checkpoint_path = os.path.join(checkpoint_dir, args_str + '.pt')

    try:
        global_graph = torch.load(os.path.join(args.data_dir, 'witg.pt'))
    except:
        build_WITG_from_trainset(datapath=args.data_dir)
        global_graph = torch.load(os.path.join(args.data_dir, 'witg.pt'))

    model = GCL4SR(args=args, global_graph=global_graph)

    train_dataset = GCL4SRData(args, train_data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True,
                                  num_workers=8)

    eval_dataset = GCL4SRData(args, valid_data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = GCL4SRData(args, test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    trainer = GCL4SR_Train(model, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        scores, result_info = trainer.eval_stage(0, test_dataloader, full_sort=True, test=True)
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        best_score = -np.inf

        for epoch in range(args.epochs):
            trainer.train_stage(epoch, train_dataloader)
            scores, _ = trainer.eval_stage(epoch, eval_dataloader, full_sort=True, test=False)
            current_score = scores[-1]

            if (epoch + 1) % 10 == 0:
                torch.save(trainer.model.state_dict(), os.path.join(checkpoint_dir, f"{args_str}_{epoch + 1}.pt"))

            if current_score > best_score:
                best_score = current_score
                early_stopping.reset()
                torch.save(trainer.model.state_dict(), args.checkpoint_path)
            else:
                early_stopping(np.array([current_score]), trainer.model)
                if early_stopping.early_stop:
                    break

        trainer.args.train_matrix = test_rating_matrix
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.eval_stage(0, test_dataloader, full_sort=True, test=True)

    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


main()
