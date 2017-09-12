# coding: utf-8

#하이퍼파라미터 최적화
#쉽게 사용 하는 방법 : 탐색 범위를 설정하고 무작위로 적용해보며 정확도를 검증데이터로 평가하는 작업을 반복하여 최적의 파라미터를 찾는다.
#신경망에서, 구조(층의 수, 뉴런의 수), 배치의 크기, 매개변수 갱신시 학습률, 가중치 감소, 등...
#훈련데이터 : 매개변수(가중치, 편향)의 학습
#검증데이터 : 하이퍼파라미터의 성능 평가
#시험데이터 : 범용 성능 평가
#하이퍼파라미터 값의 '좋음'을 시험데이터로 확인하게 되면, 하이퍼파라미터의 값이 시험 데이터에만 적합하도록 조정될 수 있음 -> 범용 성능 확인 불가
#10의 거듭제곱 형태로 범위를 지정(로그 스케일)하는 것이 효과적이다. (10**-3 ~ 10**3)
#탐색 범위가 넓다면 당연히 오랜 시간이 걸림 -> 학습을 위한 에폭을 작게하여 1회 평가에 걸리는 시간을 단축하는 것이 효과적
#사실 하이퍼파라미터는 수행자의 '지혜'와 '직관'에 의존하는 느낌이 강하다. 세련된 기법을 원한다면 베이즈 최적화를 공부해보자.

import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

#데이터 로딩
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

#결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

#20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate) #검증데이터로 추출할 데이터 개수. int형으로 바꿔줘야 아래 코드가 돌아감
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

#학습 함수 정의
def __train(lr, weight_decay, epocs=50): #탐색할 lr, weight_decay는 아래 for문을 통해서 주어지게 됨
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list



# 하이퍼파라미터 무작위 탐색======================================
#100회간 무작위로 주어진 범위에서 탐색할 수를 선택해서 탐색
optimization_trial = 100
#결과물 종합용 딕셔너리 생성
results_val = {}
results_train = {}

for _ in range(optimization_trial):
    #탐색 하이퍼 파라미터 범위 지정====================
    weight_decay = 10 ** np.random.uniform(-8, -4) #(10**-8, 10**-4) 범위 지정
    lr = 10 ** np.random.uniform(-6, -2)
    # ============================================

    #위에서 지정한 범위내 파라미터 하나가 샘플링되어 __train 함수 인자로 들어가고, 결과로 acc_list 두개를 얻는다.
    val_acc_list, train_acc_list = __train(lr, weight_decay)

    #학습 경과 확인하기. (가장 마지막 val_acc와 파라미터 조합을 보여준다)
    print("val acc:" + str(val_acc_list[-1]) + " | lr: " + str(lr) +", weight decay: " + str(weight_decay))
    #결과 저장할 딕셔너리의 키를 해당 파라미터를 정리한 문자열로 정의
    key = 'lr: ' + str(lr) + ', weight decay: ' + str(weight_decay)
    #결과 종합용 딕셔너리에 각각 넣어준다.
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list


# 그래프 그리기
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20 #20개만 그리고, col은 5개이므로 row는 알아서 개수가 정해짐
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))


#for문 들어가기 전에, sorted의 key라는 인자를 보자
student_tuples = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]
sorted(student_tuples, key=lambda student_tubles: student_tubles[2])   # sort by age
sorted(student_tuples, key=lambda x: x[2])   # student_tuples의 쌍 하나하나를 그냥 x라 해도 무관함

#validation acc
i=0
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    #result_val.items()는 type이 dict_items인데, 리스트나 튜플과 유사하다고 생각하면 된다.
    #이 안에는 각각의 dictionary 요소들이 ([key 1, item 1], [key 2, item 2], ...)형태로 묶여있다.
    #sort할 대상을 key가 정해주는데, results_val.items내의 하나의 list, 예를들어 [key 1, item 1]을 선택했다면
    #[key 1, item 1]의 [1][-1]번째 값을 대표로 세워서 비교하자는 것이다. 즉, item 1의 제일 마지막 수가 된다.
    #이는 임의의 탐색 파라미터를 통해 학습이 진행되었을 때, val_acc의 제일 마지막 값이다.
    #이를 비교하여 best model을 찾는다.

    print("Best-" + str(i+1) + "(val acc: " + str(val_acc_list[-1]) +") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)

    #제일 왼쪽 열이 아닌 대상은 y축 지표들을 없앤다
    if i % 5:
        plt.yticks([])

    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list, label='val_acc')
    plt.plot(x, results_train[key], "--", label='tr_acc')
    i += 1

    if i >= graph_draw_num:
        break

plt.legend()
plt.show()

#파라미터 탐색 결과
#상위 5개만을 봤을 때, lr은 0.004 ~ 0.007, weight decay는 7.3e-08 ~ 7.3e-05를 확인 할수 있음
#이 범위로 다시 탐색을 진행하고, 이 과정을 반복하여 좋은 하이퍼 파라미터를 찾을 수 있다.
#최종적으로 만들어진 모형에 대해 test data를 적용하여 범용 성능 평가를 한다.