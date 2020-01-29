#!/usr/bin/env python
# coding: utf-8


import os,sys,inspect
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform

import random
import train_utils
import models, models.densenet
import datasets, datasets.xray


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-name', type=str)
parser.add_argument('--output_dir', type=str, default="/lustre04/scratch/cohenjos/concept-embedding2/")
parser.add_argument('--dataset', type=str, default="chex")
parser.add_argument('--dataset_dir', type=str, default="not used yet")
parser.add_argument('--model', type=str, default="densenet121")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=160, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--threads', type=int, default=8, help='')
parser.add_argument('--taskweights', type=bool, default=True, help='')
parser.add_argument('--featurereg', type=bool, default=False, help='')
parser.add_argument('--weightreg', type=bool, default=False, help='')
parser.add_argument('--data_aug', type=bool, default=True, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')

cfg = parser.parse_args()


data_aug = None
if cfg.data_aug:
    data_aug = torchvision.transforms.Compose([
        datasets.xray.ToPILImage(),
        torchvision.transforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        torchvision.transforms.ToTensor()
    ])

transforms = datasets.xray.XRayResizer(224)

datas = []
datas_names = []
if "nih" in cfg.dataset:
    dataset = datasets.xray.NIH_XrayDataset(
        datadir="/lustre04/scratch/cohenjos/NIH/images-224",
        csvpath="/lustre03/project/6008064/jpcohen/ChestXray-NIHCC/Data_Entry_2017.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("nih")
if "pc" in cfg.dataset:
    dataset = datasets.xray.PC_XrayDataset(
        datadir="/lustre04/scratch/cohenjos/PC/images-224",
        csvpath="/lustre03/project/6008064/jpcohen/PADCHEST_SJ/labels_csv/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("pc")
if "chex" in cfg.dataset:
    dataset = datasets.xray.CheX_XrayDataset(
        datadir="/lustre03/project/6008064/jpcohen/chexpert/CheXpert-v1.0-small",
        csvpath="/lustre03/project/6008064/jpcohen/chexpert/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("chex")
if "google" in cfg.dataset:
    dataset = datasets.xray.NIH_Google_XrayDataset(datadir="/lustre04/scratch/cohenjos/NIH/images-224",
        csvpath="/lustre03/project/6008064/jpcohen/ChestXray-NIHCC/google_labels.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
if "mimic_ch" in cfg.dataset:
    dataset = datasets.xray.MIMIC_XrayDataset(
          datadir="/lustre04/scratch/cohenjos/MIMIC/images-224/files",
          csvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
          metacsvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
          transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("mimic_ch")
if "mimic_nb" in cfg.dataset:
    dataset = datasets.xray.MIMIC_XrayDataset(
          datadir="/lustre04/scratch/cohenjos/MIMIC/images-224/files",
          csvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-negbio.csv.gz",
          metacsvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
          transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("mimic_nb")
if "openi" in cfg.dataset:
    dataset = datasets.xray.Openi_XrayDataset(
            datadir="/lustre03/project/6008064/jpcohen/OpenI/images/",
            xmlpath="/lustre03/project/6008064/jpcohen/OpenI/ecgen-radiology/",
            transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
if "kaggle" in cfg.dataset:
    dataset = datasets.xray.Kaggle_XrayDataset(
            datadir="/lustre03/project/6008064/jpcohen/kaggle-pneumonia/stage_2_train_images_jpg",
            csvpath="/lustre03/project/6008064/jpcohen/kaggle-pneumonia/stage_2_train_labels.csv",
            transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("kaggle")


print("datas_names", datas_names)

for d in datas:
    datasets.xray.relabel_dataset(datasets.xray.default_pathologies, d)

#cut out training sets
for i, dataset in enumerate(datas):
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    #disable data aug
    test_dataset.data_aug = None
    
    #fix labels
    train_dataset.labels = dataset.labels[train_dataset.indices]
    test_dataset.labels = dataset.labels[test_dataset.indices]
    
    train_dataset.pathologies = dataset.pathologies
    datas[i] = train_dataset
    
if len(datas) == 0:
    raise Exception("no dataset")
elif len(datas) == 1:
    train_dataset = datas[0]
else:
    print("merge datasets")
    train_dataset = datasets.xray.Merge_XrayDataset(datas, label_concat=cfg.label_concat)


# Setting the seed
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("dataset.labels.shape", train_dataset.labels.shape)
    
# create models
if "densenet" in cfg.model:
    model = models.densenet.DenseNet(num_classes=train_dataset.labels.shape[1], in_channels=1, 
                                     **models.densenet.get_densenet_params(cfg.model)) 
elif "resnet101" in cfg.model:
    model = torchvision.models.resnet101(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
elif "shufflenet_v2_x2_0" in cfg.model:
    model = torchvision.models.shufflenet_v2_x2_0(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
else:
    raise Exception("no model")


train_utils.train(model, train_dataset, cfg)


print("Done")
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                            batch_size=cfg.batch_size,
#                                            shuffle=cfg.shuffle,
#                                            num_workers=0, pin_memory=False)






