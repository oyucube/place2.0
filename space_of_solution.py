# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
# 乱数のシード固定
#
# i = np.random()
# np.random.seed()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import chainer
from chainer import cuda, serializers
import sys
from tqdm import tqdm
import datetime
import importlib
import image_dataset
import socket
from PIL import Image, ImageDraw, ImageFont
from chainer.links import VGG16Layers


def get_batch(ds, index, repeat):
    nt = ds.num_target
    # print(index)
    batch_size = index.shape[0]
    return_x = np.empty((batch_size, 3, 256, 256))
    return_t = np.zeros((batch_size, nt))
    for bi in range(batch_size):
        return_x[bi] = ds[index[bi]][0]
        return_t[bi][ds[index[bi]][1]] = 1
    return_x = return_x.reshape(batch_size, 3, 256, 256).astype(np.float32)
    return_t = return_t.astype(np.float32)
    return_x = chainer.Variable(xp.asarray(xp.tile(return_x, (repeat, 1, 1, 1))))
    return_t = chainer.Variable(xp.asarray(xp.tile(return_t, (repeat, 1))))
    return return_x, return_t


def vgg_extract(vgg_m, ds, index, repeat):
    nt = ds.num_target
    batch_size = index.shape[0]
    return_t = np.zeros((batch_size, nt))
    return_i = np.empty((batch_size, 3, 256, 256))
    image_list = []
    for bi in range(batch_size):
        image_list.append(ds.get_vgg(index[bi]))
        return_t[bi][ds[index[bi]][1]] = 1
        return_i[bi] = ds[index[bi]][0]

    with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
        feature = vgg_m.extract(image_list, layers=["conv5_3"])["conv5_3"]

    return_t = return_t.astype(np.float32)
    return_i = return_i.reshape(batch_size, 3, 256, 256).astype(np.float32)
    return_i = chainer.Variable(xp.asarray(xp.tile(return_i, (repeat, 1, 1, 1))))
    return_x = chainer.Variable(xp.asarray(xp.tile(feature.data, (repeat, 1, 1, 1))))
    return_t = chainer.Variable(xp.asarray(xp.tile(return_t, (repeat, 1))))
    return return_x, return_t, return_i


def draw_attention(d_img, d_l_list, d_s_list, index, save="", acc=""):
    draw = ImageDraw.Draw(d_img)
    color_list = ["red", "yellow", "blue", "green"]
    size = 256
    for j in range(l_list.shape[0]):
        l = d_l_list[j][index]
        s = d_s_list[j][index]
        print(l)
        p1 = (size * (l - s / 2))
        p2 = (size * (l + s / 2))
        p1[0] = size - p1[0]
        p2[0] = size - p2[0]
        print([p1[0], p1[1], p2[0], p2[1]])
        draw.rectangle([p1[0], p1[1], p2[0], p2[1]], outline=color_list[j])
    if len(acc) > 0:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\msgothic.ttc", 20)
        draw.text([120, 230], acc, font=font, fill="red")
    if len(save) > 0:
        img.save(save + ".png")

#  引数分解
parser = argparse.ArgumentParser()
# load model id

# * *********************************************    config    ***************************************************** * #
parser.add_argument("-a", "--am", type=str, default="model_at",
                    help="attention model")
parser.add_argument("-l", "--l", type=str, default="scale",
                    help="load model name")
test_b = 100
num_step = 2
label_file = "15"

# * **************************************************************************************************************** * #

parser.add_argument("-b", "--batch_size", type=int, default=50,
                    help="batch size")
parser.add_argument("-e", "--epoch", type=int, default=50,
                    help="iterate training given epoch times")
parser.add_argument("-m", "--num_l", type=int, default=40,
                    help="a number of sample ")
parser.add_argument("-s", "--step", type=int, default=2,
                    help="look step")
parser.add_argument("-v", "--var", type=float, default=0.02,
                    help="sample variation")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="use gpu")
# train id
parser.add_argument("-i", "--id", type=str, default="sample",
                    help="data id")

# model save id
parser.add_argument("-o", "--filename", type=str, default="v1",
                    help="prefix of output file names")
args = parser.parse_args()

file_id = args.filename
model_id = args.id
num_lm = args.num_l
n_epoch = args.epoch
train_id = args.id

train_b = args.batch_size
train_var = args.var
gpu_id = args.gpu

# naruto ならGPUモード
if socket.gethostname() == "naruto":
    gpu_id = 0
    log_dir = "/home/y-murata/storage/place397/"
    train_dataset = image_dataset.ImageDataset("/home/y-murata/data_256", label_file)
    val_dataset = image_dataset.ValidationDataset("/home/y-murata/val_256", label_file)
else:
    log_dir = ""
    train_dataset = image_dataset.ImageDataset(r"C:\Users\waka-lab\Documents\place365\data_256", label_file)
    val_dataset = image_dataset.ValidationDataset(r"C:\Users\waka-lab\Documents\place365\val_256", label_file)

xp = cuda.cupy if gpu_id >= 0 else np

data_max = train_dataset.len
test_max = val_dataset.len
img_size = 256
n_target = train_dataset.num_target
num_class = n_target
target_c = ""
if test_b > test_max:
    test_b = test_max

# モデルの作成
model_file_name = args.am
if model_file_name.find("vgg") != -1:
    vgg = True
    vgg_model = VGG16Layers()
else:
    vgg = False
sss = importlib.import_module("modelfile." + model_file_name)
model = sss.SAF(n_out=n_target, img_size=img_size, var=train_var, n_step=num_step, gpu_id=gpu_id)
# model load
if len(args.l) != 0:
    print("load model model/best{}{}.model".format(args.l, label_file))
    serializers.load_npz('model/best' + args.l + label_file + '.model', model)
else:
    print("load model!!!")
    exit()

# gpuの設定
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()


nd = xp.array(range(100))
sample = 100
det = 10
space = xp.zeros((sample, det, det))
for s in tqdm(range(sample)):
    for i in range(det):
        l2 = xp.array([[0.5, i / det]]).astype("float32")
        for j in range(det):
            s2 = xp.array([[j / det]]).astype("float32")
            with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
                x, t = get_batch(val_dataset, xp.array([nd[s]]), 1)
                space[s][i][j] = model.s2_determin(x, t, l2, s2)
                # print("y:{} s:{} acc:{}".format(i / det, j/det, space[s][i][j]))


# 描画
for i in range(sample):
    plt.figure()
    plt.xlabel("Size")
    plt.ylabel("position Y")
    plt.pcolor(space[i], cmap=plt.cm.Blues)
    plt.colorbar()
    plt.savefig("buf/space" + str(i) + ".png")
    plt.close()

with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
    x, t = get_batch(val_dataset, nd[0:sample], 1)
    acc, l_list, s_list = model.use_model(x, t)
print(acc)

for i in range(sample):
    save_filename = "buf/attention" + str(i)
    # print(acc[i])
    img = val_dataset.get_image(nd[i])
    acc_str = ("{:1.8f}".format(acc[i]))
    print(acc_str)
    print(acc[i])
    draw_attention(img, l_list, s_list, i, save=save_filename, acc=acc_str[0:6])
