# coding: utf-8

#lena.png에 적용해보기
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from ch07_CNN.ex02_simple_convnet import SimpleConvNet
from matplotlib.image import imread
from common.layers import Convolution


def filter_show(filters, nx=4, show_num=16):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(show_num / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(show_num):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')


network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'filter_pad': 0, 'filter_stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 학습된 가중치 불러오기
network.load_params("./ch07_CNN/params.pkl")
# 학습된 가중치 ploting
filter_show(network.params['W1'], 16)

#lena_gray.png 이미지 불러오기
img = imread('./dataset/lena_gray.png')
img = img.reshape(1, 1, *img.shape) # *를 해주면 튜플 자체를 untuple 해서 넣어준다.

fig = plt.figure()

w_idx = 1

for i in range(16):
    w = network.params['W1'][i]
    b = 0  # network.params['b1'][i]

    w = w.reshape(1, *w.shape)
    # b = b.reshape(1, *b.shape)
    conv_layer = Convolution(w, b)
    out = conv_layer.forward(img)
    out = out.reshape(out.shape[2], out.shape[3])

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
#각각의 필터에 따라 다른 엣지나 블롭을 찾는 것을 확인할 수 있다.
