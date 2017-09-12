#곱셈 layer
class MulLayer:
    def __init__(self): #x, y를 초기화
        self.x = None
        self.y = None

    def forward(self, x, y): #입력받은 x, y를 저장하면서 out을 계산
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

#역전파
dprice = 1 #맨 위의 loss에서 흘러 내려갈 gradient 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dapple_price)



#덧셈 layer
class AddLayer:
    def __init__(self): #덧셈은 input을 기억할 필요가 없음
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_tax_layer = MulLayer()
add_apple_orange_layer = AddLayer()

#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)

#역전파
dprice = 1 #맨 위의 loss에서 흘러 내려갈 gradient, 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dorange, dorange_num, dall_price, dtax)