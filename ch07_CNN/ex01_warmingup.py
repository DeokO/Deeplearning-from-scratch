# coding: utf-8

#im2col 구현
import sys, os
sys.path.append(os.path.pardir)
from common.util import im2col
import numpy as np

x1 = np.random.rand(1, 3, 7, 7) # 7 x 7의 3채널을 가진 데이터 1개
col1 = im2col(x1, 5, 5, stride=1, pad=0) #input data인 x1을 stride=1이고, padding=0인 상황에서 5 x 5에 3채널을 가진 filter에 적용하기 위해 전개
print(col1.shape) #(9, 75) 적용할 filter에 대해 본 input data에서는 총 9번 연산을 해야하고, 한 벡터의 길이는 75(= 5 * 5 * 3)이다.

x2 = np.random.rand(10, 3, 7, 7) #batch 단위로 10개의 데이터를 한번에 input
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) #(90, 75) 총 연산 횟수가 단지 10배 더 됨


#im2col을 이용하여 ConvLayer 구현하기
class Convolution:
    def __init__(self, W, b, stride=1, pad=0): #필터, 편향, 스트라이드, 패딩
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x): #x : input data
        #filter의 형태(FN: 필터 개수, C: 채널 수, FH: 필터 높이, FW:필터 너비)
        FN, C, FH, FW = self.W.shape
        #input data의 형태
        N, C, H, W = x.shape
        #출력될 feature map의 높이, 너비
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        #im2col 적용 부분
        col = im2col(x, FH, FW, self.stride, self.pad) #input을 im2col로 전개
        col_W = self.W.reshape(FN, -1).T #filter를 reshape함. -1이 있는 열 부분은 FN에 의해 알아서 결정되도록 해라 라는 의미이다.
        out = np.dot(col, col_W) + self.b #행렬곱 연산 후 bias 더해줌

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        #먼저 out으로 나온 값을 reshape를 통해 (N, height, width, ?)형태로 바꾼다. 이때 ?는 사실 채널과 같다.
        #np의 transpose를 이용해서 축의 순서를 바꾼다. (N, height, width, C)를 (N, C, height, width)순으로 바꿔줌
        #(N, height, width, C) -> (N, C, height, width)
        # 0,   1,      2,   3  ->  0, 3,   1,      2             : 여기의 0, 3, 1, 2를 순서로 이용

        return out



#CNN에서 im2col을 이용하는 경우 역전파시 im2col을 역으로 처리해야 한다.
#common/util.py의 col2im을 이용하면 됨.
#col2im을 이용한다는 점을 제외하면 합성곱 계층의 역전파는 Affine 계층과 똑같다.
#합성곱 계층의 역전파 구현은 common/layer.py에 있다.





#im2col을 이용하여 PoolingLayer 구현하기
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0): #풀링 필터의 height, weight, 스트라이드, 패딩
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x): #x : input data
        #input data의 형태
        N, C, H, W = x.shape
        #출력될 feature map의 높이, 너비
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        #im2col 적용 부분
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) #input을 im2col로 전개
        col = col.reshape(-1, self.pool_h*self.pool_w)

        #행별 최대값 구하기
        out = np.max(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out