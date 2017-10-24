from model_dram import DRAM
from chainer import Variable
from env import xp
import make_sampled_image
import chainer.functions as F
import chainer.links as L


class SAF(DRAM):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
        super(DRAM, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            glimpse_cnn_1=L.Convolution2D(3, 32, 3),  # in 32 out 30
            glimpse_cnn_2=L.Convolution2D(32, 64, 3),  # in 30 out 28
            glimpse_cnn_3=L.Convolution2D(64, 128, 3),  # in 14 out 12
            glimpse_cnn_4=L.Convolution2D(128, 128, 3),  # in 12 out 10
            glimpse_full=L.Linear(5 * 5 * 128, n_units),
            glimpse_loc=L.Linear(2, n_units),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1),

            l_norm_c1=L.BatchNormalization(32),
            l_norm_c2=L.BatchNormalization(64),
            l_norm_c3=L.BatchNormalization(128),
            l_norm_c4=L.BatchNormalization(128),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),

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
        self.gsize = 32
        self.train = True
        self.var = 0.015
        self.vars = 0.015
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5
        self.n_step = n_step

    def recurrent_forward(self, xm, lm):
        hgl = F.relu(self.glimpse_loc(Variable(lm.data)))

        hg1 = F.relu(self.l_norm_c1(self.glimpse_cnn_1(Variable(xm))))
        hg2 = F.relu(self.l_norm_c2(self.glimpse_cnn_2(hg1)))
        hg3 = F.relu(self.l_norm_c3(self.glimpse_cnn_3(F.max_pooling_2d(hg2, 2, stride=2))))
        hg4 = F.relu(self.l_norm_c4(self.glimpse_cnn_4(hg3)))
        hgf = F.relu(self.glimpse_full(F.max_pooling_2d(hg4, 2, stride=2)))

        hr1 = F.relu(self.rnn_1(hgl * hgf))
        # ベクトルの積
        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, y, b

    def make_img(self, x, l, num_lm, random=0):
        s = xp.log10(xp.ones((1, 1)) * 64 / 256) + 1
        sm = xp.repeat(s, num_lm, axis=0)

        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            lm = xp.clip(l.data + eps * xp.sqrt(self.vars), 0, 1)
            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_rgb_gpu(lm.data, sm, x.data, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm_rgb(lm.data, sm, x.data, num_lm, g_size=self.gsize)
        return xm, lm
