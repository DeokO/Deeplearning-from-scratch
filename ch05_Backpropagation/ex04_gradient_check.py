import sys, os
sys.path.append(os.path.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch04_Neural_Network_Training.two_layer_net import TwoLayerNet
from common.layers import *

#데이터 로딩 / 배치 추출
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_batch = x_train[:3]
t_batch = t_train[:3]

#network 구성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#수치미분과 해석적 미분 값을 모두 구해서 비교
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
