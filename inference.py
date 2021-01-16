#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File:inference.py

Created on 2020/8/9

@author: danlan
"""

import os
import csv
import torch
import argparse
import pandas as pd
import torch.nn as nn
import numpy as np
import utils.utility as utility

from importlib import import_module
from data.common import list_pictures
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torch.utils.data import dataloader

from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking

parser = argparse.ArgumentParser(description='MGN')

parser.add_argument('--nThread', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')

parser.add_argument("--datadir", type=str, default="../dataset/data_PRCV2020", help='dataset directory')
parser.add_argument('--data_train', type=str, default='Market1501', help='train dataset name')
parser.add_argument('--data_test', type=str, default='Market1501', help='test dataset name')

parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--test_every', type=int, default=20, help='do test per every N epochs')
parser.add_argument("--batchtest", type=int, default=32, help='input batch size for test')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--model', default='MGN', help='model name')

parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--pool', type=str, default='avg', help='pool function')
parser.add_argument('--feats', type=int, default=256, help='number of feature maps')
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--num_classes', type=int, default=906, help='')  # train:906 test: 389

parser.add_argument("--margin", type=float, default=0.3, help='')
parser.add_argument("--random_erasing", action='store_true', help='')
parser.add_argument("--probability", type=float, default=0.5, help='')

parser.add_argument("--savedir", type=str, default='saved_models', help='directory name to save')
parser.add_argument("--outdir", type=str, default='out', help='')
parser.add_argument("--resume", type=int, default=-1, help='resume from specific checkpoint')
parser.add_argument('--save', type=str, default='test', help='file name to save')
parser.add_argument('--load', type=str, default='MGN_adam', help='file name to load')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')

args = parser.parse_args()


def saveCSV(filename, data_dict):
    assert filename is not None
    assert data_dict is not None
    is_csv = filename.find('.csv')
    assert is_csv > 0
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key_q in data_dict.keys():
            write_list = []
            write_list.append(key_q)
            for key_g, value in data_dict[key_q].items():
                write_list.append(key_g)
                write_list.append(value)
            writer.writerow(write_list)


class Model(nn.Module):
    def __init__(self, args, ckpt):
        super(Model, self).__init__()
        print('[INFO] Making model...')

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.nGPU = args.nGPU
        self.save_models = args.save_models

        # import mgn
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        if not args.cpu and args.nGPU > 1:
            self.model = nn.DataParallel(self.model, range(args.nGPU))

        self.load(
            ckpt.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        if self.nGPU == 1:
            return self.model
        else:
            return self.model.module

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        print('apath:', apath)
        print(resume)
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )


class RAP(dataset.Dataset):

    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader
        print('reading data !!!!!!')

        data_path = args.datadir
        if dtype == 'query':
            data_path += '/query_images'
        elif dtype == 'gallery':
            data_path += '/test_images'

        self.imgs = [path for path in list_pictures(data_path)]  # img path list
        print(dtype + ' images num:', len(self.imgs))

    def __getitem__(self, index):
        path = self.imgs[index]


        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)


class Tester():
    def __init__(self, args, model, testset, queryset, test_loader, query_loader, ckpt):
        self.args = args
        self.test_loader = test_loader
        self.query_loader = query_loader

        self.testset = testset
        self.queryset = queryset

        self.ckpt = ckpt
        self.model = model
        self.device = torch.device('cpu' if args.cpu else 'cuda')

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        for inputs in loader:
            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            for i in range(2):
                if i == 1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)  # torch.Size([-1, 2048])
        return features

    def test(self):

        self.model.eval()
        qf = self.extract_feature(self.query_loader).numpy()
        gf = self.extract_feature(self.test_loader).numpy()

        print('query shape:', qf.shape)
        print('gallery shape:', gf.shape)

        # 查看提取的图像特征是否归一化
        # print(qf[0])
        # sum_qf = []
        # sum_gf = []
        # for i in range(len(qf)):
        #     sum_qf.append(np.sum(qf[i] ** 2))
        # for i in range(len(gf)):
        #     sum_gf.append(np.sum(gf[i] ** 2))
        #
        # print('sum_qf length:', len(sum_qf))
        # print('sum_gf length:', len(sum_gf))

        # 计算距离
        # if self.args.re_rank:
        #     q_g_dist = np.dot(qf, np.transpose(gf))
        #     q_q_dist = np.dot(qf, np.transpose(qf))
        #     g_g_dist = np.dot(gf, np.transpose(gf))
        #     dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        # else:
        dist = cdist(qf, gf)
        print(dist.shape)

        # 结果保存为CSV文件
        res_df = pd.DataFrame(dist)

        data_path = args.datadir
        query_data_path = data_path + '/query_images'
        gallery_data_path = data_path + '/test_images'

        query_inx_list = [path.rsplit('_', 1)[1].split('.')[0] for path in list_pictures(query_data_path)]
        gallery_inx_list = [path.rsplit('_', 1)[1].split('.')[0] for path in list_pictures(gallery_data_path)]
        res_df.index = query_inx_list
        res_df.columns = gallery_inx_list

        res_dict_dictlist = {}
        for i in res_df.index:
            res_dict_i = pd.DataFrame(res_df.loc[i]).sort_values(by=[i], axis=0, ascending=False).to_dict()
            res_dict_dictlist.update(res_dict_i)

        saveCSV('csv_test.csv', res_dict_dictlist)

        # evaluate
        # r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
        #         separate_camera_set=False,
        #         single_gallery_shot=False,
        #         first_match_break=True)
        # m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        # self.ckpt.log[-1, 0] = m_ap
        # self.ckpt.log[-1, 1] = r[0]
        # self.ckpt.log[-1, 2] = r[2]
        # self.ckpt.log[-1, 3] = r[4]
        # self.ckpt.log[-1, 4] = r[9]
        # best = self.ckpt.log.max(0)
        # self.ckpt.write_log(
        #     '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
        #     m_ap,
        #     r[0], r[2], r[4], r[9],
        #     best[0][0],
        #     (best[1][0] + 1)*self.args.test_every
        #     )
        # )
        # print(
        #     '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} )'.format(
        #     m_ap,
        #     r[0], r[2], r[4], r[9],
        #     )
        # )


if __name__ == '__main__':
    ckpt = utility.checkpoint(args)

    # load model
    model = Model(args, ckpt)

    # load data
    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = RAP(args, test_transform, 'gallery')
    queryset = RAP(args, test_transform, 'query')

    test_loader = dataloader.DataLoader(testset, batch_size=args.batchtest, num_workers=args.nThread)
    query_loader = dataloader.DataLoader(queryset, batch_size=args.batchtest, num_workers=args.nThread)

    # predict distance
    tester = Tester(args, model, testset, queryset, test_loader, query_loader, ckpt)
    tester.test()
