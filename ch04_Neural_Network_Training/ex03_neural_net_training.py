# coding: utf-8

#MNIST 이용, 2층 신경망 학습

import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from common.functions import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x: 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) #input x와 target t가 주어졌을때, W로 loss를 미분하고자 함

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) #common.gradient.py에 있는 함수를 사용함
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    #5장에서 구현할 내용이지만, 미리 당겨와서 사용한다.
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
net.params['W1'].shape #(784, 100)
net.params['b1'].shape #(100, )r
net.params['W2'].shape #(100, 10)
net.params['b2'].shape #(10, )

#이미지 데이터를 아무렇게나 100장 만든다.
x = np.random.rand(100, 784) #더미 입력 데이터 100개
t = np.random.rand(100, 10) #더미 정답 레이블 100개
#gradient 구하기
grads = net.numerical_gradient(x, t) #기울기 계산. 현재 방법으로는 시간이 너무 오래 걸린다.
grads['W1'].shape
grads['b1'].shape
grads['W2'].shape
grads['b2'].shape



#########################################################################################################
# 미니 배치로 구현하기
#########################################################################################################
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch04_Neural_Network_Training.two_layer_net import TwoLayerNet #본 dir의 two_layer_net에 정의된 TwoLayerNet을 이용한다.

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#하이퍼 파라미터
iters_num = 10000 #반복횟수
train_size = x_train.shape[0]
batch_size = 100 #배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

#1epoch 당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
for i in range(iters_num):
    #경과 확인
    # print(i)

    #미니 배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) #성능 개선한 함수

    #매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]

    #학습 경과 기록
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

    #1epoch당 acc를 기록
    if i % iter_per_epoch == 0: #60000 / 100 회만큼 돌아야 1번의 epoch라 할 수 있고, 이때마다 아래를 실행
        train_acc = network.accuracy(x_batch, t_batch)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

#시각화
import matplotlib.pyplot as plt
plt.plot(train_acc_list, 'b', label='train_acc')
plt.plot(test_acc_list, 'r', label='test_acc', linestyle='--')
plt.legend()
plt.show()

