from chainer import serializers
import chainer
from chainer.links import VGG16Layers
from PIL import Image

model = VGG16Layers()


img = Image.open("00000001.jpg")

img1 = Image.open("00000002.jpg")

with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
    feature = model.extract([img, img1, img1], layers=["conv5_3"])["conv5_3"]

print(feature.data.shape)
