# coding: utf-8

#Dropout : 뉴런을 임의로 삭제하면서 학습하는 방법
#신경망 모델이 복잡해지면 가중치 감소만으로는 대응하기 어려워짐
#훈련 때 은닉층의 뉴런을 무작위로 골라 삭제하여 신호를 전달하지 않음
#시험 때는 모든 뉴런에 신호를 전달하는데, 뉴런의 출력에 학습 때 삭제한 비율을 곱하여 dropout으로 인한 감소를 보정해줌
#그런데, 현재는 위와같이 시험 때 곱해주는 것이 아니라, 학습 때 dropout 비율인 p를 나눠서 weight 자체를 수정해서 학습함
#dropout을 적용하면 훈련데이터와 시험데이터에 대한 정확도 차이가 줄음(overfit 방지 / 표현력 향상)
#앙상블 : 개별적으로 학습시킨 여러 모델의 출력을 평균내어 추론하는 방식
#드롭아웃이 학습 때 뉴런을 무작위로 삭제하는 행위를 매번 다른 모델을 학습시키는 것으로 해석한다면, 앙상블 학습과 같은 효과를 (대략)하나의 네트워크로 구현했다고 생각할 수 있음

import sys, os
sys.path.append(os.path.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

#데이터 로딩
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.2
# ====================================================

#dropout의 비율을 0.2로 사용한 모형 객체 생성
do_network = MultiLayerNetExtend(input_size=784,
                                 hidden_size_list=[100, 100, 100, 100, 100, 100],
                                 output_size=10,
                                 use_dropout=use_dropout,
                                 dropout_ration=dropout_ratio)

#dropout을 사용하지 않은 모형 객체 생성
network = MultiLayerNetExtend(input_size=784,
                                 hidden_size_list=[100, 100, 100, 100, 100, 100],
                                 output_size=10,
                                 use_dropout=False)

#dropout의 비율을 0.2로 사용한 모형 학습 객체 생성
do_trainer = Trainer(do_network, x_train, t_train, x_test, t_test,
                     epochs=301, mini_batch_size=100, optimizer='sgd',
                     optimizer_param={'lr': 0.01}, verbose=True) #verbose는 진행사항을 보여줄지 설정

#dropout을 사용하지 않은 모형 학습 객체 생성
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                     epochs=301, mini_batch_size=100, optimizer='sgd',
                     optimizer_param={'lr':0.01}, verbose=True)

#두 모형 학습 진행
do_trainer.train()
trainer.train()

#두 모형에서 학습시의 정확도 기록과 테스트시의 정확도 기록을 list로 받아옴
do_train_acc_list, do_test_acc_list = do_trainer.train_acc_list, do_trainer.test_acc_list
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기
#dropout 사용
markers = {'train': 'o', 'test': 's'}
plt.subplot(1,2,1)
x = np.arange(len(do_train_acc_list))
plt.plot(x, do_train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, do_test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.title('Use dropout=0.2')
plt.legend(loc='lower right')

#dropout 사용하지 않음
plt.subplot(1,2,2)
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.title('No dropout')
plt.legend(loc='lower right')

plt.show()

