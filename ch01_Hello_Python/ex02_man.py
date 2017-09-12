# coding: utf-8

#Class를 정의하면 개발자가 독자적인 자료형을 만들 수 있음.
#클래스만의 전용 함수와 속성을 정의할 수 있음

class Man: #Man(object) 라고 하면 object 클래스를 상속 받음을 의미한다.

    #생성자 : 클래스가 객체화될때 처음에만 불러지며, 초기화 진행한다.
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    #메서드(=함수)
    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David") #<__main__.Man at 0x223734ea438> main부의 Man이라는 클래스를 의미함.
m.hello()
m.goodbye()