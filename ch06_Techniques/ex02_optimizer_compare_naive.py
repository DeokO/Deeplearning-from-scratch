# coding: utf-8


import sys, os
sys.path.append(os.path.pardir)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

#공통으로 비교할 함수 정의
def f(x, y):
    return x**2 / 20.0 + y**2

#위 함수 f를 편미분한 결과를 출력
def df(x, y):
    return x/10.0, 2.0*y

#초기값 및 parameter, gradient 설정
init_pos = (-7., 2.)
#이동해가면서 찍힐 위치
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
#초기 gradient
grads = {}
grads['x'], grads['y'] = 0, 0

#optimizer를 4개를 이용해서 비교해 보자.
#SGD, Momentum, AdaGrad, Adam
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.95)
optimizers['Momentum'] = Momentum(lr=0.1)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

#sub plot용 idx
idx = 1


#각 optimizer에 대해 돌아가면서 plot을 그린다.
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    #update를 해나가면서 history를 30개 찍어두는 부분
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)

    #등고선을 그리기 위해 x축, y축으로 움직일 point를 설정
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    #x, y를 엮어서 좌표 형태로 만들어준다.
    X, Y = np.meshgrid(x, y)
    #각 좌표를 f함수에 넣어서 등고선의 값이 될 Z값을 산출한다.
    Z = f(X, Y)

    # 등고선 외곽선 단순화
    mask = Z > 7
    Z[mask] = 0

    #ploting
    plt.subplot(2, 2, idx) #(2, 2) 형태의 subplot을 그릴거고, 그중에 idx번째 plot임을 명시
    plt.plot(x_history, y_history, 'o-', color='red') #line을 o- 형태로 빨간색으로 그림
    plt.contour(X, Y, Z) #배경으로 등고선을 그려줌
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+') #center에 +형태를 찍어줌
    plt.title(key) #현재 optimizer의 이름을 title로 줌
    plt.xlabel('x')
    plt.ylabel('y')
    idx += 1  # 다음 subplot의 idx를 표시

plt.show()
