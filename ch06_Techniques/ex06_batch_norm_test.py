# coding: utf-8

#Batch normalize
#각 층이 활성화 값을 적당히 퍼뜨리도록 강제하기
#장점 :  1. 학습을 빨리 진행할 수 있다.(학습 속도 개선)
#       2. 초기값에 크게 의존하지 않는다.
#       3. 오버피팅을 억제한다.(드랍아웃 등의 필요성 감소)


# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

#데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

#weight 초기화를 std기준으로 바꿔가면서 network를 생성하고자 함. 네트워크 구축을 함수로 정의해둠
def __train(weight_init_std):
    #batch normalize 사용한 network
    bn_network = MultiLayerNetExtend(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100, 100],
                                     output_size=10,
                                     weight_init_std=weight_init_std,
                                     use_batchnorm=True)
    # batch normalize 사용하지 않은 network
    network = MultiLayerNetExtend(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100, 100],
                                     output_size=10,
                                     weight_init_std=weight_init_std,
                                     use_batchnorm=False)
    optimizer = SGD(lr=learning_rate)

    #train data의 accuracy를 담을 리스트 생성
    bn_train_acc_list = []
    train_acc_list = []

    #1회 epoch당 진행될 batch 학습 횟수 설정
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    #10억번 반복동안
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        #각각의 네트워크에 배치를 전달하고, 기울기를 구하고 최적화 기법(SGD)를 이용해서 update 진행
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        #1회 epoch가 진행될때마다 아래의 내용을 실행
        if i % iter_per_epoch == 0:
            #1 에폭 마다 bn_network와 bn을 사용하지 않은 network의 학습 acc를 기록 / 출력한다.
            bn_train_acc = bn_network.accuracy(x_batch, t_batch)
            bn_train_acc_list.append(bn_train_acc)
            train_acc = network.accuracy(x_batch, t_batch)
            train_acc_list.append(train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    #최종적으로 출력하는 것은 bn을 사용하지 않은 network의 학습시 acc와 bn을 사용한 network의 학습시 acc를 담은 리스트를 출력한다.
    #실제로 딥러닝을 사용하게 된다면, 이 정보도 물론 중요하지만, 모델 자체를 저장해야 할 것이다.
    return train_acc_list, bn_train_acc_list


# 그래프 그리기
#weight의 std용으로 16개의 후보를 만듦
weight_scale_list = np.logspace(0, -4, num=16) #log를 취했을 때 0부터 -4까지 되는 수 16개의 seq를 얻어냄
#플롯의 x축 역할
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    #총 16개의 실험 중 어디까지 진행되었는지 보는 출력
    print("============== " + str(i+1) + "/16" + "===============")
    #for문에서 받은 weight의 std 하나를 학습 함수에 넣어서 학습을 하고, bn을 사용하지 않은 network의, 사용한 network의 acc list를 받아옴
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(4, 4, i+1)
    plt.title("W: " + str(w))

    #마지막 번째 플롯일 경우
    if i == 15:
        plt.plot(x, bn_train_acc_list, label="Batch Normalization", markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', label="Normal(without BatchNorm)", markevery=2)
    #마지막 번째 플롯이 아닌 경우
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    # 제일 왼쪽이 아닌(나눠서 1, 2, 3의 정수가 나오는 == True) 경우에는 y축의 marker를 지워준다.
    if i % 4:
        plt.yticks([])
    # 제일 왼쪽의 subplot들은 ylabel을 accuracy라고 줌
    else:
        plt.ylabel('accuracy')

    # 제일 아래줄이 아닌 경우에는 x축의 marker를 지워준다.
    if i < 12:
        plt.xticks([])
    # 제일 아래줄의 subplot들은 xlabel을 epochs라고 줌
    else:
        plt.xlabel("epochs")

    plt.legend(loc='lower right')

plt.show()