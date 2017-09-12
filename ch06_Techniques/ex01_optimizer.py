# coding: utf-8

import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads): #params, grads는 dict 형태. key에 W1, b1, W2, b2등이 들어 있을 것임
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9): #learning rate=0.01, 관성은 0.9정도를 줌
        self.momentum = momentum
        self.lr = lr
        self.v = None #관성으로 나갈 방향을 초기화

    def update(self, params, grads):
        if self.v is None:  #관성으로 나갈 방향이 초기화되어있다면,
            self.v = {}     #dictionary 형태로 데이터를 만드는데,
            for key, val in params.items():
                self.v[key] = np.zeros_like(val) #params의 형태마다 각각 0으로 초기화 해준다.

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


#Momentum 개념과 유사한데, Momentum은 이번 grads를 구하고, 현재 가진 관성방향과 같이 움직이도록 하는데 반해
#NAG(Nesterov's Accelerated Gradient)은 먼저 관성 방향으로 움직인 뒤, grads를 구해서 거기에서 gradient 방향으로 움직인다.
#관성에 좀 더 힘을 주어 일단 먼저 갔던 속도만큼 가본다음에, gradient를 계산해서 방향을 조정하겠다는 것이다.
# v_(t+1) <- momentum * v_(t) - lr * grad(W_t + momentum * v_t) #현재 관성 방향으로 움직이고, 움직인 위치에서 grad 구해서 움직임
class Nesterov:
    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    # NAG는 모멘텀에서 한 단계 발전한 방법이다. (http://newsight.tistory.com/224)

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


#Adagrad는 러닝레이트를 다음과 같은 노멀라이제이션 L2 term으로 나누어주는 것이다.
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None       #learning rate의 감소할 양을 각 축별로 고려해주기 위한 부분

    def update(self, params, grads):
        if self.h is None:  #축소할 양이 초기화되어 있는 경우
            self.h = {}     #dictionary 형태로 초기화
            for key, val in params.items():
                self.h[key] = np.zeros_like(val) #params의 형태마다 각각 0으로 초기화 해준다.

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h + 1e-7)


#AdaGrad는 진행될 수록 h(lr 줄이는 부분)가 점점 커져서 결국에는 update되는 부분이 0이 되게 된다.
#RMSprop은 lr을 줄이는 양을 단순히 제곱값을 계속 더해나가지 않고, 원래 있던 것을 조금 줄이면서(* 0.99), 새로 들어온 것을 약간 더해준다.
#h <- decay_rate * h + (1 - decay_rate) * grads**2
#W <- W - lr * grads(W) / (sqrt(h) + 1e-7)
class RMSprop:
    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


#Momentum과 AdaGrad를 합침
#하이퍼 파라미터의 편향 보정이 진행되는 특징을 가짐
#m = beta1*m + (1-beta1)*dx #위 Momentum의 v 역할을 함. 원래 방향을 유지한 채로, 새 update 방향을 약간 더해줌
#v = beta2*v + (1-beta2)*(dx**2) 위 AdaGrad의 h 역할을 함. lr을 조금씩 줄여나가는 역할.
#x += - learning_rate * m / (np.sqrt(v) + eps)
class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
