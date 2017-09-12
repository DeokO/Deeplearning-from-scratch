# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

#나쁜 구현 예
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x))/h
# 위의 h는 반올림 오차 발생
np.float32(1e-50)
# 차분에서 오차가 있음 f(x+h) - f(x) -> 중앙 차분을 이용한다.
def numerical_diff(f, x):
    h = 1e-4 #이 정도면 괜찮다고 소개하고 있음
    return (f(x+h) - f(x-h)) / (2*h)

#수치 미분의 예
#y = 0.01 * x**2 + 0.1 * x
def function_1(x):
    return 0.01 * x**2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()

numerical_diff(function_1, 5) #실제로는 0.2이 나와야 정상
numerical_diff(function_1, 10) #실제로는 0.3이 나와야 정상

#접선까지 그리기
def tangent_line(f, x):
    d = numerical_diff(f, x) #기울기
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y #어떤 람다식을 return함

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y) #곡선
plt.plot(x, y2) #직선
plt.show()

def function_2(x):
    return x[0]**2 + x[1]**2
    #또는 return np.sum(x**2)

#기울기 (gradient)
def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size): #각 축별로 계산
        tmp_val = x[idx]
        #f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

_numerical_gradient_no_batch(function_2, np.array([3., 4.])) #원래는 6.00000037801, 7.99999999991189 이런 형태지만 numpy가 약간 보기 쉽게 보정해서 출력함
_numerical_gradient_no_batch(function_2, np.array([0., 2.]))
_numerical_gradient_no_batch(function_2, np.array([3., 0.]))


#시각화 해보기
def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

#배치단위로 구할 경우
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad
grad = numerical_gradient(function_2, np.array([X, Y]))

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444") #gradient의 -값을 표현함.
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()



#경사 하강법 구현
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad

    return x

init_x = np.array([-3., 4.])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100) #거의 (0,0)에 가까운 수치
gradient_descent(function_2, init_x=init_x, lr=10, step_num=100) #발산
gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=10) #수렴하기 전에 끝남


################################################################################################################
# weight에 대한 손실함수의 기울기 구하기
################################################################################################################
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) #정규분포에서 추출한 난수로 (2,3)형태의 matrix를 생성

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

#객체 생성
net = simpleNet()
print(net.W)

#new input을 이용해 예측
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p)

#실제 label과의 loss 계산
t = np.array([0, 1, 0])
net.loss(x, t)

#W가 인자로 들어가는 함수를 하나 생성해 둠
def f(W):
    return net.loss(x, t)

#f(net의 loss)를 net.W로 편미분
dW = numerical_gradient(f, net.W)

#람다 기법을 사용하면 더 간편하게 구현 가능함.
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)


