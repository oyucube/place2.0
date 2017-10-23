from model_dram import DRAM
from chainer import Variable
from env import xp
import make_sampled_image


class SAF(DRAM):
    def make_img(self, x, l, num_lm, random=0):
        s = xp.log10(xp.ones(1) * 40 / 256) - 1
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
