# coding: utf-8

#계단 함수 구현하기
import numpy as np
def step_function(x):
    y = x > 0
    return y.astype(np.int)
x = np.array([-1., 1., 2.])
step_function(x)

#계단 함수 그래프
import matplotlib.pyplot as plt
def step_function(x):
    return np.array(x>0, dtype=np.int)
x = np.arange(-5., 5., 0.1) #-5부터 5까지 0.1 단위로 seq 생성
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

#시그모이드 함수 구현하기
def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.array([-1., 1., 2.])
sigmoid(x)

#시그모이드 함수 그래프
x = np.arange(-5., 5., 0.1) #-5부터 5까지 0.1 단위로 seq 생성
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

#두 함수 비교 그래프
x = np.arange(-5., 5., 0.1) #-5부터 5까지 0.1 단위로 seq 생성
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1, label='step-function')
plt.plot(x, y2, label='sigmoid', linestyle='--')
plt.ylim(-0.1, 1.1)
plt.show()

#ReLU(Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)
