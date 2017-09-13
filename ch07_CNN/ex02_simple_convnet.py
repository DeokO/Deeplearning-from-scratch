# coding: utf-8


import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


#CNN network 구현
class SimpleConvNet:
    """Simple CNN

    구조 : conv - relu - pool - affine - relu - affine - softmax

    Parameters=================================================
    intput_size : 입력 데이터의 크기(MNIST의 경우 784)
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트
    output_size : 출력 크기(MNIST의 경우 10)
    activation : 활성화 함수 - 'relu' 또는 'sigmoid'
    weight_init_std : 가중치의 표준편차 설정(e.g. 0.01)
       'relu'나 'he'로 지정하면 'He 초기값'으로 설정
       'sigmoid'나 'xavier'로 지정하면 'Xavier 초기값'으로 설정
    ===========================================================
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'filter_pad': 0, 'filter_stride':1}, #conv layer가 하나여서 conv_param이 list형태로 들어오지 않아도 됨
                #input의 pad는 0이고, 1칸씩 stride하면서 filter의 형태는 5 x 5이고, 이것이 30개 있는 상황
                 hidden_size=100, output_size=10, weight_init_std=0.01):
                #Affine layer에서 노드 수는 100개, 출력은 10개 클래스, 가중치 초기화는 0.01 표준편차


        #self.이 없는 변수들은 필드로써 저장이 되지 않아서 나중에 빼서 볼 수 없다.
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['filter_pad']
        filter_stride = conv_param['filter_stride']
        input_size = input_dim[1] #input 이미지의 한 변

        #conv layer 통과 후 한변의 길이
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        #2x2 pooling layer 통과 후 원소의 개수
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        #가중치 초기화
        self.params = {}
        #conv layer의 파라미터
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size) #(FN, C, FH, FW)
        self.params['b1'] = np.zeros(filter_num)
        #pooling layer의 파라미터
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        #affine layer의 파라미터
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        #계층 생성
        self.layers = OrderedDict()
        #conv layer 생성.
        #filter의 형태(4차원)와 bias, stride, padding을 전달한다.
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['filter_stride'], conv_param['filter_pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2) #2x2 pooling
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    #input x가 들어오고 forward로 쭉 나가는 for문
    def predict(self, x):
        for layer in self.layers.values():
            #앞서 들어오는 x를 각 layer를 거치며 output을 다시 x에 저장해서 다시 또 input으로 사용
            #softmax loss 직전까지만 하므로, 최종 결과 값은 softmax 들어가기 전인 score까지만 계산한다.

            #각 layer에서 사용된 class들(Convolution, Relu, Pooling, Affine)에는 각각 forward, backward 메소드가 정의되 있음
            x = layer.forward(x)

        return x

    #input x와 target t가 들어오면 loss를 구한다.
    def loss(self, x, t):
        """손실 함수를 구한다.
                Parameters
                ----------
                x : 입력 데이터
                t : 정답 레이블
                """
        y = self.predict(x)
        return self.last_layer.forward(y, t) #last_layer가 softmax cross entropy 구하는 layer임. y, t가 이때 last_layer의 필드로 저장됨

    #데이터 x, t에 대해 accuracy 구한다.
    def accuracy(self, x, t, batch_size=100):
        #만약 target이 one-hot vector처럼 2차원 이상이라면, 각 행별로 argmax를 구해줌
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        #제대로 맞힌 개수들을 더해주는 부분
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            #배치사이즈 단위로 데이터를 자르고 tx, tt에 저장
            tx = x[i * batch_size:(i+1) * batch_size]
            tt = t[i * batch_size:(i+1) * batch_size]

            #tx에 대해 예측
            y = self.predict(tx)
            y = np.argmax(y, axis=1)

            #정확히 맞힌 개수 합치기
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    #수치 미분으로 얻은 각 층의 가중치와 편향의 gradient를 구함
    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """

        #w(weight)만 결정되면 loss(x(인자로 주어짐), t(인자로 주어짐))를 구하는 간단한 람다식을 정의
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3): #3개 층에 대해 아래 코드 진행
            #numerical_gradient(f[funcion], x[parameter])
            #예) numerical_gradient(loss_w, W1)인 경우, loss_w함수에 대해 W1의 현재 값에서 약간의 값(h=1e-7)을 더하고 빼면서 그 변화는 양을 기울기로 구한다
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        #처음 전파되는 기울기는 1
        dout = 1
        dout = self.last_layer.backward(dout) # 10(output)의 크기의 기울기 벡터
        #역전파는 역순으로 for문에서 backward 진행
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            #실행 동안 'Conv1', 'Affine1', 'Affine2'의 layer에서는 gradient가 필드로 저장되고, 전파시킬 gradient만 출력한다.
            #relu, pool등은 그냥 전파할 내용만 출력한다.

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    #학습한 파라미터셋(가중치, 편향)을 피클 데이터로 저장
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items(): #저장해온 필드 params 딕셔너리를 param이라는 딕셔너리에 옮겨담음
            params[key] = val
        with open(file_name, 'wb') as f: #write모드로 파일을 f라는 이름으로 열어두고
            pickle.dump(params, f) #f에 params 딕셔너리를 덤프

    #저장한 파라미터셋 로딩
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f: #read모드로 파일을 f라는 이름으로 열어두고
            params = pickle.load(f) #load

        for key, val in params.items():
            self.params[key] = val #해당 객체의 필드에다가 모두 저장해줌

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

