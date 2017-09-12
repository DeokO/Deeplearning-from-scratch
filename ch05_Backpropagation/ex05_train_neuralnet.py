# coding: utf-8

import sys, os

from ch03_Neural_Network.ex03_MNIST_example_and_batch import t_test

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch04_Neural_Network_Training.two_layer_net import TwoLayerNet

#데이터 로딩 / 배치 추출
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#하이퍼 파라미터 설정
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

#지표 확인용 list
train_loss_list = []
train_acc_list = []
test_acc_list = []

#1epoch당 iter 횟수 설정
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #오차 역전파법으로 기울기를 구한다
    grad = network.gradient(x=x_batch, t=t_batch)

    #갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch, t_batch)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
