# coding: utf-8

#MNIST : 28*28의 회색조 이미지이며, 각 픽셀은 0부터 255의 값을 가짐. 숫자 label이 붙어있음. 학습데이터 60,000장, 시험 데이터 10,000장
import sys, os
sys.path.append(os.pardir) #부모 디렉터리의 파일을 가져오기 위해 path를 지정해둔다.
from dataset.mnist import load_mnist

#처음 한번은 몇 분 정도 걸림
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) #http://yann.lecun.com/exdb/mnist/ 이 사이트에서 데이터를 받아와야 하는데 이거 자체가 먹통인 상황.

#각 데이터의 형상 출력
print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000, )
print(x_test.shape) #(10000, 784)
print(t_test.shape) #(10000, )

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image #python image library 모듈

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
label = t_train[0]
print(label) #5

print(img.shape)    # (784, )
img = img.reshape(28, 28) #원래 이미지 모양으로 변형
print(img.shape)    # (28, 28)

img_show(img) #이미지 열기


#신경망의 추론 처리
#input : 784 차원 -> output : 10 차원
#은닉층 h1 : 50개 뉴런, h2 : 100개 뉴런
import pickle, os

#data load function
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)

    return x_test, t_test

#network initialize
os.chdir('./ch03_Neural_Network')  # 현재 dir 설정 (sample_weight.pkl 불러오기 위함)
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

#activation-sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

#activation-softmax
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)

    return exp_a/sum_exp_a

#예측된 y 산출
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

#정확도 평가
# x, t = get_data()
x, t = x_test, t_test
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    print(i)
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))


#구조 확인
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
x.shape
print(x[0].shape, W1.shape, W2.shape, W3.shape,)

#배치 처리
x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size): #0부터 데이터 개수번째까지, batch_size 단위로 커지면서 seq 생성
    print(i)
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) #y_batch.shape : (100, 10). 행단위로 최대 arg를 output으로 생각하자. 열 전체를 보고 argmax 구함: axis=1, 행 전체를 보고 argmax 구함: axis=0
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) #boolean을 np.sum 하면 int형으로 더해짐

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))


