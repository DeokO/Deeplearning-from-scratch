# coding: utf-8

#numpy 배열 클래스인 numpy.array에는 행렬, 배열 연산에 편리한 메서드가 많이 준비되 있음
import numpy as np

#배열 생성시 np.array() 메서드 이용
x = np.array([1., 2., 3.])
print(x)
type(x)

#산술 연산 (원소 개수가 같은 경우, element-wise operate)
x = np.array([1., 2., 3.])
y = np.array([2., 4., 6.])
x+y
x-y
x*y
x/y
x / 2.0 #브로드캐스트

#numpy의 N차원 배열
A = np.array([[1, 2], [3, 4]])
print(A)
A.shape #np.shape(A)와 같음
A.dtype
B = np.array([[3, 0], [0, 6]])
A+B
A*B #원소 곱
np.dot(A, B) #행렬 곱\
A*10

#브로드캐스트: 크기가 맞지 않을 때, 작은 형태를 가지는 것을 n배 하여 큰 형태가 되게 만들고 연산
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
A*B

#원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
X[0] #1행
X[0][1] #1행 2열 원소

#X에 대해 iterate 관점으로 접근하면 row단위로 output을 제공한다.
for row in X:
    print(row)

#인덱스를 배열로 지정해 한번에 여러 원소에 접근할 수도 있다.
X = X.flatten() #하나의 벡터로 쭉 펼침
print(X)
X[np.array([0, 2, 4])] #X[[0, 2, 4]] 이렇게 해도 됨. list 형태나 array 형태나 가능
#위 기능으로 15초과 원소를 찾을 수 있음
X>15
X[X>15]


#python같은 동적 언어는 C, C++같은 정적 언어보다 처리속도가 느림. 빠른 성능이 요구될 경우, C, C++로 구현하는 것을 추천하며,
#python은 이때는 중개자의 역할을 한다. numpy도 주된 처리는 C, C++로 구현되어 있어서, 성능을 해치지 않으면서 python의 편리한 문법을 사용 가능하게 함.