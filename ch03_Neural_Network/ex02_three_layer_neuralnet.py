# coding: utf-8

#numpy의 다차원 배열
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
np.ndim(A) #1차원 (벡터)
A.shape #벡터의 길이가 4. 4행인 열벡터라고 생각하면 될 듯 하다.
A.shape[0]

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
np.ndim(B)
B.shape

#행렬의 내적
A = np.array([[1, 2], [3, 4]])
A.shape
B = np.array([[5, 6], [7, 8]])
B.shape
np.dot(A, B)

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
np.dot(A, B)
C = np.array([[1, 2], [3, 4]])
#np.dot(A, C) 행렬의 곱의 규칙을 만족 못하므로, 오류가 발생함. 첫번째 행렬의 차원1의 원소수 3 != 두번째 행렬의 차원0의 원소수 2

#브로드캐스팅
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([7, 8])
np.dot(A, B)


#신경망의 내적
#input: 2차원, W: 2차원을 3차원으로 mapping, Y: 3차원 output
X = np.array([1, 2])
print(X); X.shape
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W); W.shape
Y = np.dot(X, W)
print(Y)

#3층 신경망 구현하기
def sigmoid(x):
    return 1/(1+np.exp(-x))
#(1층)
X = np.array([1., .5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([.1, .2, .3])
print(X.shape); print(W1.shape); print(B1.shape)
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
#(2층)
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape); print(W2.shape); print(B2.shape)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
#(3층, 출력층)
def identity_function(x):
    return x
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

#구현 정리
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([.1, .2, .3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)


#softmax function
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3, 2.9, 4.0])
softmax(a)
#softmax의 input으로 너무 큰 값이 들어가면 overflow 문제가 발생 할 수 있다. 이럴 때는 약간의 보정을 해줘서 softmax가 계산되게 해준다.
a = np.array([1010, 1000, 990])
#np.exp(a) / np.sum(np.exp(a)) #너무 큰 값이 들어가서 overflow 발생
c = np.max(a)
np.exp(a-c) / np.sum(np.exp(a-c)) #max값을 빼는 형태로 input을 넣어주면 각 score간의 차이는 보존되면서 올바르게 계산된다.
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #overflow 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
