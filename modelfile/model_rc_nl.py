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
