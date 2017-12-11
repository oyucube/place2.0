# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""
import chainer.functions as F
from chainer import Variable
from env import xp
import make_sampled_image
from modelfile.model_bnlstm import BaseBN


class SAF(BaseBN):
    def __call__(self, x, target, mode):
        self.reset()
        n_step = self.n_step
        num_lm = x.data.shape[0]
        if mode == 1:
            r_buf = 0
            l, s, b = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=2)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)

                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
                    r = xp.where(
                        xp.argmax(y.data, axis=1) == xp.argmax(target.data, axis=1), 1, 0).reshape((num_lm, 1)).astype(
                        xp.float32)

                    loss += F.sum((r - b) * (r - b))
                    k = self.r * (r - b.data)
                    loss += F.sum(k * r_buf)

                    return loss / num_lm
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)
                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
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

    def make_img(self, x, l, s, num_lm, random=0):
        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
            sm = Variable(xp.clip(s.data, 0, 1))
        elif random == 1:
            lm = Variable(xp.ones((num_lm, 2)).astype(xp.float32) * 0.5)
            sm = Variable(xp.ones((num_lm, 1)).astype(xp.float32))
        else:
            eps = xp.random.rand(num_lm, 2).astype(xp.float32)
            epss = xp.random.rand(num_lm, 1).astype(xp.float32)
            sm = Variable(epss)
            lm = Variable(eps)
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_rgb_gpu(lm.data, sm.data, x.data, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm_rgb(lm.data, sm.data, x.data, num_lm, g_size=self.gsize)
        return xm, lm, sm
