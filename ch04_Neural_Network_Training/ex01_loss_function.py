# coding: utf-8

#평균 제곱 오차 MSE
import numpy as np
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
#정답 2
t = [0,0,1,0,0,0,0,0,0,0]
#2일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t)) #0.0975 #그냥 mean_squared_error(y, t)하면 적용이 되지 않음. list간의 -가 정의되지 않았기 때문

#7일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t)) #0.5975
#위에서 label을 잘 맞춘 경우가 loss가 더 낮음을 확인할 수 있음.


#교차 엔트로피 오차 CEE
#수식에 의해, 정답 위치에 해당하는 y의 log를 취한 값이 해당 row에서 loss가 됨
def cross_entropy_error(y, t):
    delta = 1e-7 #log 0을 막는 역할. -inf가 되는 것을 막음
    return -np.sum(t * np.log(y + delta))
#정답 2
t = [0,0,1,0,0,0,0,0,0,0]
#2일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t)) #0.5108

#7일 확률이 가장 높다고 추정
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t)) #2.3025
#위에서 label을 잘 맞춘 경우가 loss가 더 낮음을 확인할 수 있음.


#미니배치 학습
from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

#무작위로 10장만 빼기
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size, replace=False) #60000미만의 수 중에서 무작위로 10개를 뽑기. replace=False를 하면 비복원추출이 됨
x_batch = x_train[batch_mask] #batch_mask를 index로 사용하여 데이터를 추출
t_batch = t_train[batch_mask]

#배치단위 교차 엔트로피 구하기
def cross_entropy_error(y, t):
    if y.ndim == 1: #벡터로 들어온 경우. batch size=1
        #t와 y 모두 2차원 형태로 바꿔줌
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

#target이 one-hot-encoding이 아닌 경우
def cross_entropy_error(y, t):
    if y.ndim == 1: #벡터로 들어온 경우. batch size=1
        #t와 y 모두 2차원 형태로 바꿔줌
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

