# coding: utf-8

#gradient를 수치미분과 해석적으로 해서 확인해보기
import numpy as np
from ch07_CNN.ex02_simple_convnet import SimpleConvNet

network = SimpleConvNet(input_dim=(1,10, 10),
                        conv_param = {'filter_num':10, 'filter_size':3, 'filter_pad':0, 'filter_stride':1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)

#임의로 (10, 10, 1)의 이미지 데이터셋 1개를 생성하고, target을 1로 한다.
X = np.random.rand(100).reshape((1, 1, 10, 10))
T = np.array([1]).reshape((1,1))

#수치 미분과 해석 미분을 각각 진행한다.
grad_num = network.numerical_gradient(X, T)
grad = network.gradient(X, T)

#수치 비교
for key, val in grad_num.items():
    print(key, np.abs(grad_num[key] - grad[key]).mean())
#큰 차이는 아님을 확인할 수 있다.