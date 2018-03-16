# wget https://www.dropbox.com/s/3z11cpntwuasm43/08127_image.png?dl=0; mv 08127_image.png?dl=0 08127_image.png 
import caffenet
import caffe
import torch
import numpy as np
from torch.autograd import Variable


def load_image(imgfile):
    height, width = 700, 700    
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, height, width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104., 117., 123.]))
    transformer.set_raw_scale('data', 7.2801098892805181)
    transformer.set_channel_swap('data', (2, 1, 0))
    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, height, width)
    return image



    net = caffenet.CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    image = Variable(image)
    blobs = net(image)
    return blobs, net.models


protofile = '/models/testpy_val_91_500_pkg.prototxt'
weightfile = '/models/test2.caffemodel'
imgfn = '08127_image.png'


image = torch.from_numpy(load_image(imgfn))
image = Variable(image)
net = caffenet.CaffeNet(protofile)
net.load_weights(weightfile)
# net = caffenet.CaffeNet(protofile)
# print(net)
# net.load_weights(weightfile)
# net.eval()

blobs = net(image)
