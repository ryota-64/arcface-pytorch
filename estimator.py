import argparse
import time
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.nn import DataParallel
from torchvision import datasets, transforms, models

from sklearn.manifold import TSNE
from sklearn import preprocessing
import umap
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as offline
# offline.init_notebook_mode()

from data.make_file_names import load_criteria_list
from data.dataset import Dataset
from config.config import Config
from models import *


parser = argparse.ArgumentParser(description='metric estimator')
parser.add_argument('--train-data-path', type=str, default='data/Datasets/invoices/train/',
                    help='input train data path (default: data/Datasets/invoices/train/)')
parser.add_argument('--estimate-data-path', type=str, default='data/Datasets/invoices/test/',
                    help='input estimate data path (default: data/Datasets/invoices/test/)')
parser.add_argument('--load-weight-path', type=str, default='checkpoints/resnet18_40.pth',
                    help='load weight data path (default: checkpoints/resnet18_40.pth')

args = parser.parse_args()
if torch.cuda.is_available():  # GPUが利用可能か確認
    device = 'cuda'
else:
    device = 'cpu'
args.device = device


def main():
    opt = Config(os.getcwd())
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))
    model.to(device)
    model.eval()
    global args

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    centroid_map = create_centroid(model, trainloader)

    test_dataset = Dataset(opt.test_root, opt.test_list, phase='test', input_shape=opt.input_shape)
    test_loader = data.DataLoader(test_dataset,
                                 batch_size=1,
                                  # batch_size=opt.test_batch_size,
                                 shuffle=True,
                                 num_workers=opt.num_workers)

    estimate(model, test_loader, centroid_map)


def estimate(model, data_loader, centroid_map):
    accs = AverageMeter()
    pred_list = list()
    labels = list()
    acc_individual = {}
    opt = Config(os.getcwd())
    criteria_list = load_criteria_list(opt.train_root)
    # 特徴量の抽出 batch_sizeは１
    for i, (imgs, pids) in enumerate(data_loader):
        imgs = imgs
        outputs = model(imgs)
        # feature_gpu = torch.cat((feature_gpu, outputs.data), 0)
        temp_labels = [criteria_list[pid] for pid in pids]
        labels.extend(temp_labels)
        # count = feature_gpu.size(0)
        outputs = outputs.detach().numpy()
        res_vec = centroid_map[:, 1:] - np.tile(outputs, (centroid_map.shape[0], 1))
        res_vec = res_vec.astype(np.float32)

        # もっとも近いセントロイドを選ぶ
        dist = pairwise_distance(torch.from_numpy(res_vec))

        pred_ans = centroid_map[np.argmin(dist.detach().numpy())][0]
        # print(pred_ans)
        pred_list.append(pred_ans)

    # 正解ラベルと照合

    acc = accuracy(pred_list, labels)
    acc_individual = accuracy_indivisual(pred_list, labels)
    print(acc/len(pred_list))

    for key, value in acc_individual.items():
        print('key: {}  {}/{}  {}%\n'.format(key, value.count(True), len(value), value.count(True) / len(value)*100))

    print('only_train_label\n')
    for key, value in acc_individual.items():
        if key in centroid_map:
            print('key: {}  {}/{}  {}%'.format(key, value.count(True), len(value), value.count(True) / len(value) * 100))
            accs.update(value.count(True) / len(value), len(value))
    print(accs)


def create_centroid(model, data_loader):

    feature_cpu = torch.FloatTensor()
    feature_gpu = torch.FloatTensor().cuda() if args.device == 'cuda' else torch.FloatTensor()

    trans_inter = 1e4
    labels = list()
    end = time.time()
    opt = Config(os.getcwd())
    criteria_list = load_criteria_list(opt.train_root)
    print(criteria_list)

    for i, (imgs, pids) in enumerate(data_loader):
        imgs = imgs
        outputs = model(imgs)
        feature_gpu = torch.cat((feature_gpu, outputs.data), 0)
        temp_labels = [criteria_list[pid] for pid in pids]
        labels.extend(temp_labels)
        count = feature_gpu.size(0)
        print('Extract Features: [{}/{}]\t'
              # 'Time {:.3f} ({:.3f})\t'
              # 'Data {:.3f} ({:.3f})\t'
              .format(i + 1, len(data_loader),
                      # batch_time.val, batch_time.avg,
                      # data_time.val, data_time.avg
                      ))
        if count > trans_inter or i == len(data_loader) - 1 :
            # print(feature_gpu.size())
            # data_time.update(time.time() - end)
            # end = time.time()
            # print('transfer to cpu {} / {}'.format(i+1, len(data_loader)))
            feature_cpu = torch.cat((feature_cpu, feature_gpu.cpu()), 0)
            feature_gpu = torch.FloatTensor().cuda() if args.device == 'cuda' else torch.FloatTensor()
            # batch_time.update(time.time() - end)
            end = time.time()
        del outputs
    # concate_array = np.concatenate([feature_cpu, np.array(labels).reshape((-1, 1))], axis=1)
    # print(concate_array.shape)
    # print(concate_array.shape[1] - 1)
    #
    # df = pd.DataFrame(concate_array)
    df = pd.DataFrame(feature_cpu.numpy())
    df['labels'] = pd.DataFrame(np.array(labels).reshape((-1, 1)))
    df_group_mean = df.groupby('labels', as_index=False).mean()
    print(df_group_mean)
    return df_group_mean.values


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return 'avg : {}\tnum : {}'.format(self.avg, self.count)


def accuracy(preds, labels):
    accs = list()
    for i in range(len(preds)):
        accs.append(preds[i] == labels[i])
    print(accs.count(True))
    return accs.count(True)


def accuracy_indivisual(preds, labels):
    acc_data = {}
    for i, label in enumerate(labels):
        if label not in acc_data:
            acc_data[label] = [preds[i] == label]
        else:
            acc_data[label].append(preds[i] == label)

    return acc_data


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def pairwise_distance(features, metric=None):
    n = features.shape[0]
    # print(features)
    # normalize feature before test
    # x = normalize(features)
    x = features
    # print(4*'\n', x.size())
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    # print(dist.size())
    # dist = dist.expand(n, n)
    # dist = dist + dist.t()
    # dist = dist - 2 * torch.mm(x, x.t())
    dist = torch.sqrt(dist)
    return dist


if __name__ == '__main__':
    main()
