# coding: utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #계층 생성
        self.layers = OrderedDict() #순서가 있는 dictionary 생성
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1']) #layers 파일에 있는 Affine 클래스
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayers = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():  #layers라는 Ordered dictionary의 value들(layer)에 대해 x를 forward 시키고 output을 다시 x에 저장
                x = layer.forward(x)        #모든 layer에는 forward와 backward가 정의되있다.
        return x

    # x: 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayers(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: #배치단위일 경우에 대비
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

    def gradient(self, x, t):
        #순전파
        self.loss(x, t)

        #역전파
        dout = 1
        dout = self.lastLayer.backward(dout) #softmax with cross entropy error 파트에서의 dout을 구함

        layers = list(self.layers.values())
        layers.reverse() #역전파시에는 순서를 거꾸로 해서 전파해야 한다.
        for layer in layers:
            dout = layer.backward(dout)

        #결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
