#numpy 배열을 인수로 받는다 가정하고 코드 작성
import numpy as np

#Relu function
class Relu:
    def __init__(self):
        self.mask = None #input이 0보다 작거나 같은지 확인하는 부분. T, F로 구성된 array

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout * 1
        return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)
mask = (x<=0)
print(mask)


#Sigmoid function
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


#배치용 Affine 계층
X_dot_W = np.array([[0,0,0], [10, 10, 10]])
B = np.array([1, 2, 3])
X_dot_W
X_dot_W + B #브로드 캐스트에 의해 각 축마다 더해짐 -> 역전파 때는 각 데이터의 역전파 값이 편향의 원소에 모여야 함

dY = np.array([[1, 2, 3], [4, 5, 6]])
dY
dB = np.sum(dY, axis=0) #열 별로(축 별로) 합
dB

#Affine layer
class Affine:
    def __init__(self, W, b):
        self.b = b
        self.W = W
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


#Softmax-with-Loss 계층 구현
from common.functions import *
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 손실
        self.y = None       # softmax의 출력
        self.t = None       # 정답 레이블(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 배치 개수로 나눠서 데이터 1개당 오차를 앞 계층으로 전파하는 점에 주의. 해당 batch의 평균적인 gradient를 이용해서 학습함
        dx = (self.y - self.t) / batch_size
        return dx

