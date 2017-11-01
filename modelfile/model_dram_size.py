from model_dram import DRAM
from chainer import Variable
from env import xp
import make_sampled_image


class SAF(DRAM):
    def use_model(self, x, t):
        self.reset()
        num_lm = x.data.shape[0]
        n_step = self.n_step
        s_list = xp.ones((n_step, num_lm, 1)) * (40 / self.img_size)
        l_list = xp.empty((n_step, num_lm, 2))
        l, b1 = self.first_forward(x, num_lm)
        for i in range(n_step):
            if i + 1 == n_step:
                xm, lm = self.make_img(x, l, num_lm, random=0)
                l1, y, b = self.recurrent_forward(xm, lm)
                l_list[i] = l1.data
                accuracy = y.data * t.data
                return xp.sum(accuracy, axis=1), l_list, s_list
            else:
                xm, lm = self.make_img(x, l, num_lm, random=0)
                l1, y, b = self.recurrent_forward(xm, lm)
            l = l1
            l_list[i] = l.data
        return

    def make_img(self, x, l, num_lm, random=0):
        s = xp.log10(xp.ones((1, 1)) * 40 / 256) + 1
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


class SAF(DRAM):
    def use_model(self, x, t):
        self.reset()
        num_lm = x.data.shape[0]
        n_step = self.n_step
        s_list = xp.ones((n_step, num_lm, 1)) * (80 / self.img_size)
        l_list = xp.empty((n_step, num_lm, 2))
        l, b1 = self.first_forward(x, num_lm)
        for i in range(n_step):
            if i + 1 == n_step:
                xm, lm = self.make_img(x, l, num_lm, random=0)
                l1, y, b = self.recurrent_forward(xm, lm)
                l_list[i] = l1.data
                accuracy = y.data * t.data
                return xp.sum(accuracy, axis=1), l_list, s_list
            else:
                xm, lm = self.make_img(x, l, num_lm, random=0)
                l1, y, b = self.recurrent_forward(xm, lm)
            l = l1
            l_list[i] = l.data
        return

    def make_img(self, x, l, num_lm, random=0):
        s = xp.log10(xp.ones((1, 1)) * 80 / 256) + 1
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