# coding: utf-8

#matplotlib : 시각화 라이브러리
#단순한 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt
#데이터 준비
x = np.arange(0, 6, 0.1) #0부터 6까지 0.1 단위로 증가하면서 seq 생성
y = np.sin(x)
#그래프 그리기
plt.plot(x, y)
plt.show()

#pyplot의 기능
import numpy as np
import matplotlib.pyplot as plt
#데이터 준비
x = np.arange(0, 6, 0.1) #0부터 6까지 0.1 단위로 증가하면서 seq 생성
y1 = np.sin(x)
y2 = np.cos(x)
#그래프 그리기
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos', linestyle='--') #cos 함수는 점선으로 그리기
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin & cos')
plt.legend() #범례 위치를 지정하지 않으면 best로 설정됨. 범례는 label이라고 지정한 것이 표현됨
plt.show()

#이미지 표시하기
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
os.chdir('./ch01_Hello_Python')
img = imread('lena.png')

plt.imshow(img)
plt.show()