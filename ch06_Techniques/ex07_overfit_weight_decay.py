# coding: utf-8

#Overfitting : 신경망이 훈련데이터에만 지나치게 적응되어 그 외의 데이터에는 제대로 대응하지 못하는 상태
#범용 성능을 지향하는 기계학습에서, 아직 보지못한 데이터에 대해서도 바르게 식별해내는 모델이 나오기를 희망함
#복잡하고 표현력이 높은 모델을 만들 수 있지만, 그만큼 오버피팅을 억제하는 기술이 중요함
#오버피팅은 주로 매개변수가 많고 표현력이 높은 모델에서, 훈련데이터가 적은 경우에서 일어난다.


#일부러 오버피팅 일으키기
#300개의 데이터만 사용, 7층 네트워크 사용, 각 층은 100개 뉴런, 활성화 함수는 Relu
#이 코드에서는 오버피팅을 막기 위한 방법으로 가중치 감소라는 방법론을 소개하고 있다.
#idea : 큰 값을 가진 weight는 학습 과정에서 그에 상응하는 페널티를 부과하여 오버피팅을 억제하는 방법(weight값이 커서 오버피팅 발생하는 경우가 많기 때문)
#즉, 페널티인 0.5 * lambda * W**2를 loss에 더해줘서 W가 커지는 것을 막는 역할을 하게 해준다.(Ridge reg.의 방식)

# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# 데이터 로딩
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay(가중치 감쇠) 설정 ==========================
# weight_decay_lambda = 0 #weight decay를 사용하지 않을 경우
weight_decay_lambda = 0.1
# =======================================================

# 과도하게 깊은 신경망 구축하기
network = MultiLayerNet(input_size=784,
                        hidden_size_list=[100, 100, 100, 100, 100, 100],
                        output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)

#하이퍼 파라미터 설정
max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

#정확도 담을 리스트 생성
train_loss_list = []
train_acc_list = []
test_acc_list = []

#1회 에폭 당 돌아갈 배치 횟수
iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    #배치 데이터 추출
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #기울기를 구하고, SGD를 이용해서 parameter를 update
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    #각 epoch마다 acc를 list에 저장 및 출력
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs: #200회 epoch이 될때까지 학습
            break


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

#train_acc만 많이 올라가고 test_acc는 낮은 현상을 볼 수 있다.
#오버피팅이 발생했다고 생각할 수 있다.