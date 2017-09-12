# coding: utf-8


import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *

# 0. MNIST 데이터 로딩
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

# 1. 실험용 설정 셋팅
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

#network, loss를 저장할 dictionary를 설정
networks = {}
train_loss = {}

#각 optimizer마다 network를 MultiLayerNet을 이용해서 똑같은 구조로 만들고, train_loss 딕셔너리를 초기화 한다.
for key in optimizers.keys():
    networks[key] = MultiLayerNet(input_size=784,
                                  hidden_size_list=[100, 100, 100, 100],
                                  output_size=10)
    train_loss[key] = []

# 2. 훈련 시작
for i in range(max_iterations):
    #4개의 최적화 기법에 똑같이 들어갈 batch 생성
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch) #배치를 넣어서 각 네트워크의 기울기를 구함
        optimizers[key].update(networks[key].params, grads) #네트워크의 parameter를 기울기에 대해 update함
        loss = networks[key].loss(x_batch, t_batch) #사실 이것이 먼저 계산되어야 하지만, 이 코드에서는 기록용으로 저장
        train_loss[key].append(loss) #각 최적화 기법의 학습 loss 리스트에 저장

    #학습 진행 경과 및 각 최적화 기법에 해당하는 loss 확인
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ':' + str(loss))


# 3. 그래프 그리기
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

