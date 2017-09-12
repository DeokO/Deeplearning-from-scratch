# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.random.randn(1000, 100)  #100차원의 데이터 1000개 생성
node_num = 100                  #각 은닉층의 노드 개수
hidden_layer_size = 5           #은닉층 수
activations = {}                #활성화값 저장

#1회 forward propagate / weight의 표준편차 1로 지정
for i in range(hidden_layer_size):

    #처음 input이 아닌경우에 작동
    if i != 0:
        x = activations[i-1] #이 전 layer에서의 활성화값을 다시 x(input)로 저장

    #layer를 넘어가는 계산
    w = np.random.randn(node_num, node_num) * 1 #weight의 표준편차를 1로 지정
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z #각 층마다 100개의 노드씩 있으니, activations에는 5개 층이 1000*100의 값을 가지는 것을 볼 수 있음
plt.hist(z.flatten(), 30, range=(0, 1)) #마지막 층에서 actvate 후의 값 z가 극단적인 부분(0 또는 1)에 분포하는 것을 확인할 수 있음

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1) #1, 5개의 subplot중 i+1번째 그림
    plt.title(str(i+1) + '-layer')
    plt.hist(a.flatten(), 30, range=(0, 1)) #구간을 30개로 쪼개서 historam 그리기. 10만개의 활성화값들을 표현하게 됨
plt.show()
#각 층의 활성화 값들이 0과 1에 치우쳐 분포되어 있음. sigmoid 함수 형태때문에 이런 결과가 나오는 것임
#시그모이드 함수는 출력이 0, 1에 가까워지면 미분이 0에 가까워짐 -> 역전파의 기울기 값이 점점 작아지다가 사라짐
#gradient vanishing이 발생함
#깊은 딥러닝 신경망에서는 큰 문제일 수 있음


#1회 forward propagate / weight의 표준편차 0.01로 지정
x = np.random.randn(1000, 100)  #100차원의 데이터 1000개 생성
activations = {}                #활성화값 저장
for i in range(hidden_layer_size):

    #처음 input이 아닌경우에 작동
    if i != 0:
        x = activations[i-1] #이 전 layer에서의 활성화값을 다시 x(input)로 저장

    #layer를 넘어가는 계산
    w = np.random.randn(node_num, node_num) * 0.01 #weight의 표준편차를 0.01로 지정
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z #각 층마다 100개의 노드씩 있으니, activations에는 5개 층이 100개씩의 값을 가지는 것을 볼 수 있음

#히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1) #1, 5개의 subplot중 i+1번째 그림
    plt.title(str(i+1) + '-layer')
    plt.hist(a.flatten(), 30, range=(0, 1)) #구간을 30개로 쪼개서 historam 그리기
plt.show()
#sigmoid가 양 극단을 모두 0, 1로 줘버리니, 이번에는 조금 더 중심에 있게끔 모아보는 idea. 즉, 표준편차를 더 줄여보자.
#0.5 부근에 집중되어 있음
#기울기 소실 문제는 일어나지 않지만, 활성화값들이 치우쳤다는 것은 표현력 관점에서 큰 문제가 있는 것
#100개 노드를 쓰나 1개 노드를 쓰나 큰 차이가 없다는 것이므로 표현력이 제한되는 것을 확인할 수 있다.






#활성화 값이 고루 분포되어야 함
#Xavier Glorot, Yoshua Bengio의 논문에서 권장하는 Xavier 초기값 사용
#모든 층에 일괄적으로 같은 표준편차를 주는 것이 아니라, 활성값 결과가 정규분포 형태가 되게끔 하고싶다.
#idea : 앞층에서 오는 활성화값의 개수(앞층의 노드의 개수)가 n개라면 표준편차가 1/sqrt(n)이 되게 초기화 하자
#뒤에 갈수록 일그러지는 것에 대해서는 활성화 함수를 sigmoid 말고 tanh를 사용하면 조금 더 보정이 된다.
x = np.random.randn(1000, 100)  #100차원의 데이터 1000개 생성
activations = {}                #활성화값 저장
#Xavier init
for i in range(hidden_layer_size):

    #처음 input이 아닌경우에 작동
    if i != 0:
        x = activations[i-1] #이 전 layer에서의 활성화값을 다시 x(input)로 저장

    #layer를 넘어가는 계산
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num) #weight의 표준편차를 1/sqrt(n)로 지정
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z #각 층마다 100개의 노드씩 있으니, activations에는 5개 층이 100개씩의 값을 가지는 것을 볼 수 있음

#히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1) #1, 5개의 subplot중 i+1번째 그림
    plt.title(str(i+1) + '-layer')
    plt.hist(a.flatten(), 30, range=(0, 1)) #구간을 30개로 쪼개서 historam 그리기
plt.show()




