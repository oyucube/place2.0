# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import make_sampled_image
from env import xp
from modelfile.bnlstm import BNLSTM

from modelfile.model_at import BASE


class BaseRC(BASE):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
        super(BASE, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            glimpse_cnn_1=L.Convolution2D(3, 32, 4),  # in 20 out 16
            glimpse_cnn_2=L.Convolution2D(32, 64, 4),  # in 16 out 12
            glimpse_cnn_3=L.Convolution2D(64, 128, 4),  # in 12 out 8
            glimpse_full=L.Linear(4 * 4 * 128, n_units),
            glimpse_loc=L.Linear(3, n_units),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1),

            l_norm_c1=L.BatchNormalization(32),
            l_norm_c2=L.BatchNormalization(64),
            l_norm_c3=L.BatchNormalization(128),

            # 記憶を用いるLSTM部分
            rnn_1=BNLSTM(n_units, n_units),
            rnn_2=BNLSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),
            attention_scale=L.Linear(n_units, 1),

            # 入力画像を処理するネットワーク
            context_cnn_1=L.Convolution2D(3, 32, 3),  # 64 to 62
            context_cnn_2=L.Convolution2D(32, 64, 4),  # 31 to 28
            context_cnn_3=L.Convolution2D(64, 64, 3),  # 14 to 12
            context_full=L.Linear(12 * 12 * 64, n_units),

            l_norm_cc1=L.BatchNormalization(32),
            l_norm_cc2=L.BatchNormalization(64),
            l_norm_cc3=L.BatchNormalization(64),

            class_full=L.Linear(n_units, n_out)
        )

        #
        # img parameter
        #
        if gpu_id == 0:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.img_size = img_size
        self.gsize = 20
        self.train = True
        self.var = var
        self.vars = var
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5
        self.n_step = n_step

    def __call__(self, x, target, mode):
        self.reset()
        n_step = self.n_step
        num_lm = x.data.shape[0]
        r2 = 0
        if mode == 1:
            r_buf = 0
            l, s, b = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)

                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
                    r = xp.where(
                        xp.argmax(y.data, axis=1) == xp.argmax(target.data, axis=1), 1, 0).reshape((num_lm, 1)).astype(
                        xp.float32)
                    if i != 0:
                        distance = xp.absolute(lm.data - rl)
                        sum_s = xp.power(10, sm.data - 1) + xp.power(10, rs - 1)
                        r_a = xp.where(
                            sum_s[:, 0] < distance[:, 0], 0, 1
                        )
                        r_b = xp.where(
                            sum_s[:, 0] < distance[:, 1], 0, 1
                        )
                        r2 = (r_a * r_b * (sum_s[:, 0] - distance[:, 0]) * (sum_s[:, 0] - distance[:, 1])) \
                            .reshape((num_lm, 1)).astype(xp.float32)

                    r = r - 0.2 * r2

                    loss += F.sum((r - b) * (r - b))
                    k = self.r * (r - b.data)
                    loss += F.sum(k * r_buf)

                    return loss / num_lm
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)
                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
                rl = lm.data
                rs = sm.data
                l = l1
                s = s1
                b = b1

        elif mode == 0:
            l, s, b1 = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                    accuracy = y.data * target.data
                    return xp.sum(accuracy)
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                l = l1
                s = s1


class SAF(BaseRC):
    pass
