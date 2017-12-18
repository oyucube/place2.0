# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""
import chainer.functions as F
from chainer import Variable
from env import xp
from modelfile.model_rc import BaseRC


class SAF(BaseRC):
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
                        norm_term = xp.power(10, 2 * (sm.data - 1)) + xp.power(10, 2 * (rs - 1))[:, 0]
                        r_a = xp.where(
                            sum_s[:, 0] < distance[:, 0], 0, 1
                        )
                        r_b = xp.where(
                            sum_s[:, 0] < distance[:, 1], 0, 1
                        )
                        r2 = (r_a * r_b * (sum_s[:, 0] - distance[:, 0]) * (sum_s[:, 0] - distance[:, 1])) / norm_term \
                            .reshape((num_lm, 1)).astype(xp.float32)

                    r = r - 0.5 * r2

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

    def recurrent_forward(self, xm, lm, sm):
        ls = xp.concatenate([lm.data, sm.data], axis=1)
        hgl = F.relu(self.glimpse_loc(Variable(ls)))

        hg1 = F.relu(self.l_norm_c1(self.glimpse_cnn_1(Variable(xm))))
        hg2 = F.relu(self.l_norm_c2(self.glimpse_cnn_2(hg1)))
        hg3 = F.relu(self.l_norm_c3(self.glimpse_cnn_3(F.max_pooling_2d(hg2, 2, stride=2))))
        hgf = F.relu(self.glimpse_full(hg3))

        hr1 = F.relu(self.rnn_1(hgf))
        # ベクトルの積
        hr2 = F.relu(self.rnn_2(hr1 * hgl))
        l = F.sigmoid(self.attention_loc(hr2))
        s = F.sigmoid(self.attention_scale(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, s, y, b
