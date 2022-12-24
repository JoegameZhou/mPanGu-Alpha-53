# !/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/7/28
# @Author: 2022 PCL taoht
##################################################################

import os
import random
import argparse

# 保存抽样的文件为0.5M大小的小文件，为了转MindRecord时不同语种类型数据的充分混合
def write_file(file_dir, data, size=0.5):
    if os.path.isfile(file_dir):
        # 限制单个文件大小
        if os.path.getsize(file_dir) / (1024 * 1024) < size:
            with open(file_dir, 'a', encoding='utf-8')as sub_fl:
                for i in data:
                    sub_fl.write(i.replace('\n', '').replace('\r', '') + '\n\n')
            return file_dir
        else:
            file_new = file_dir.split('--')[0] + '--' + str(int(file_dir.split('--')[1][:-4])+1) + '.txt'
            with open(file_new, 'a', encoding='utf-8')as sub_fl:
                for i in data:
                    sub_fl.write(i.replace('\n', '').replace('\r', '') + '\n\n')
            return file_new
    else:
        with open(file_dir, 'a', encoding='utf-8')as sub_fl:
            for i in data:
                sub_fl.write(i.replace('\n', '').replace('\r', '') + '\n\n')
        return file_dir


def file_sample2_little_files(file_path, output_path='./test/', sample_size=10, size=0.5, big_size=0.5):
    """
    :param file_path:
    :param output_path: 保存路径
    :param sample_size: 提取容量，按MB计算
    :param size: 保存单文件size, MB
    :param big_size: 大文件容量设置，大文件使用迭代读取，加速和节省内存
    :return: None
    """
    prob = sample_size/(os.path.getsize(file_path) / (1024 * 1024))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(file_path, 'r', encoding="utf-8")as fl:
        # 大文件，迭代读取
        if os.path.getsize(file_path) / (1024 * 1024 * 1024) > big_size:
            train_new = []
            for idx, info_one in enumerate(fl):
                if info_one != '\n' and info_one:
                    if prob <= 1:
                        if random.randint(0, 10000) < prob*10000:
                            train_new.append(info_one)
                    else:
                        for i in range(int(prob)):
                            train_new.append(info_one)
                        if random.randint(0, 10000) < (prob-int(prob))*10000:
                            train_new.append(info_one)
                if idx %200000000 == 0 and idx != 0:
                    print("Big files process: ", idx)
        else:
            train_d = fl.read().split("\n\n")
            random.shuffle(train_d)
            train_new = []
            if prob == 1:
                train_new.extend(train_d)
            elif prob > 1:
                for i in range(int(prob)):
                    train_new.extend(train_d)
                random.shuffle(train_d)
                train_new.extend(train_d[:int((prob - int(prob))*len(train_d))])
            else:
                random.shuffle(train_d)
                train_new.extend(train_d[:int((prob - int(prob))*len(train_d))])

        train_file_name = output_path + file_path.split('/')[-1].split('.')[0] + '--0' + '.txt'
        k_sample = 200
        random.shuffle(train_new)
        print("Train sample: ", len(train_new))
        for idx in range(len(train_new) - 1):
            if idx%k_sample == 0:
                train_file_name = write_file(train_file_name, train_new[idx: idx+k_sample], size=size)

# # 设定抽样策略
# data_dir = '/cache/data/'
# save_path = '/cache/data_sample/'
# # 配置中、英单语文件抽取容量， MB
# mono_sample_strategy = {'zh': 100, 'en': 100}
# # 配置中、英双语文件抽取容量， MB
# corpus_sample_strategy = {'zh-en': 200}
#
# # 单语抽取
# python dataset_sample.py --data_path data_dir
#                          --output_path save_path
#                          --sample_strategy mono_sample_strategy
#                          --mode 'mono'
# # 双语抽取
# python dataset_sample.py --data_path data_dir
#                          --output_path save_path
#                          --sample_strategy corpus_sample_strategy
#                          --mode 'corpus'
import ast
parser = argparse.ArgumentParser(description='mPanGu dataset sample')
parser.add_argument('--data_path', type=str, default='/cache/data/')
parser.add_argument('--output_path', type=str, default='/cache/data_sample/')
parser.add_argument('--sample_strategy', type=ast.literal_eval,
                    default="{'en':100, 'zh':100}",
                    help='dict, MB')
parser.add_argument('--mode', type=str,
                    default='mono',
                    help='mono or corpus sample')
parser.add_argument('--save_file_size', type=float,
                    default=0.5,
                    help='0.5 MB, Use the default size')
args = parser.parse_args()

if __name__ == "__main__":

    # 设定抽样策略
    sample_strategy = args.sample_strategy
    print(sample_strategy)
    if args.mode == 'mono':
        mono_file = [args.data_path + i for i in os.listdir(args.data_path) if 'corpus' not in i]

        for idx, file in enumerate(mono_file):
            langs = file.split('/')[-1].split('.')[0]
            if langs in sample_strategy.keys():
                if sample_strategy[langs] != 0:
                    sample_size = sample_strategy[langs]
                    print("Process, idx:", idx, " langs:", langs, " , sample_size:", sample_size, "MB , file_dir:", file.split('/')[-1])
                    file_sample2_little_files(file_path=file, output_path=args.output_path, sample_size=sample_size, size=args.save_file_size)
    else:
        corpus_file = [args.data_path + i for i in os.listdir(args.data_path) if 'corpus' in i]

        for idx, file in enumerate(corpus_file):
            langs = file.split('/')[-1].split('_')[0]
            if langs in sample_strategy.keys():
                if sample_strategy[langs] != 0:
                    sample_size = sample_strategy[langs]
                    print("Process, idx:", idx, " langs:", langs, " , sample_size:", sample_size, "MB , file_dir:", file.split('/')[-1])
                    file_sample2_little_files(file_path=file, output_path=args.output_path, sample_size=sample_size, size=args.save_file_size)




