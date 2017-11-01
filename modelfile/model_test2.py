from modelfile.model_at import BASE
from chainer import Variable
from env import xp
import make_sampled_image
import chainer.functions as F
import chainer.links as L
from modelfile.bnlstm import BNLSTM


class SAF(BASE):
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
            baseline=L.Linear(2 * n_units, 1),

            l_norm_c1=L.BatchNormalization(32),
            l_norm_c2=L.BatchNormalization(64),
            l_norm_c3=L.BatchNormalization(128),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(2 * n_units, 2),
            attention_scale=L.Linear(2 * n_units, 1),

            # 入力画像を処理するネットワーク
            context_cnn_1=L.Convolution2D(3, 32, 3),  # 64 to 62
            context_cnn_2=L.Convolution2D(32, 64, 4),  # 31 to 28
            context_cnn_3=L.Convolution2D(64, 64, 3),  # 14 to 12
            context_full=L.Linear(12 * 12 * 64, n_units),

            l_norm_cc1=L.BatchNormalization(32),
            l_norm_cc2=L.BatchNormalization(64),
            l_norm_cc3=L.BatchNormalization(64),

            class_full=L.Linear(2 * n_units, n_out)
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
        self.var = 0.015
        self.vars = 0.015
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = n_step
        self.n_step = n_step
        self.whole_image = None

    def first_forward(self, x, num_lm):
        h2 = F.relu(self.l_norm_cc1(self.context_cnn_1(F.average_pooling_2d(x, 4, stride=4))))
        h3 = F.relu(self.l_norm_cc2(self.context_cnn_2(F.max_pooling_2d(h2, 2, stride=2))))
        h4 = F.relu(self.l_norm_cc3(self.context_cnn_3(F.max_pooling_2d(h3, 2, stride=2))))
        h4r = F.relu(self.context_full(h4))
        self.whole_image = h4r
        hl = F.relu(self.rnn_1(h4r))
        h5 = F.concat((self.whole_image, hl), axis=1)

        l = F.sigmoid(self.attention_loc(h5))
        s = F.sigmoid(self.attention_scale(h5))
        b = F.sigmoid(self.baseline(Variable(h5.data)))
        return l, s, b

    def recurrent_forward(self, xm, lm, sm):
        ls = xp.concatenate([lm.data, sm.data], axis=1)
        hgl = F.relu(self.glimpse_loc(Variable(ls)))

        hg1 = F.relu(self.l_norm_c1(self.glimpse_cnn_1(Variable(xm))))
        hg2 = F.relu(self.l_norm_c2(self.glimpse_cnn_2(hg1)))
        hg3 = F.relu(self.l_norm_c3(self.glimpse_cnn_3(F.max_pooling_2d(hg2, 2, stride=2))))
        hgf = F.relu(self.glimpse_full(hg3))

        hrl = F.relu(self.rnn_1(hgl * hgf))

        hr2 = F.concat((self.whole_image, hrl), axis=1)

        # ベクトルの積
        l = F.sigmoid(self.attention_loc(hr2))
        s = F.sigmoid(self.attention_scale(hr2))
        y = F.softmax(self.class_full(hr2))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, s, y, b

