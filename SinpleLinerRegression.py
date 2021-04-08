import numpy as np
import matplotlib.pyplot as plt
while True:
    x_arr = input("Please enter the value of x for training：")
    x_train = np.array([float(n) for n in x_arr.split()])
    y_arr = input("Please enter the value of y for training：")
    y_train = np.array([float(n) for n in y_arr.split()])
    if x_train.shape != y_train.shape:
        print("The quantity of x you entered does not match that of y. Please re-enter it.")
    if x_train.shape == y_train.shape:
        break
x_ave = np.mean(x_train)
y_ave = np.mean(y_train)





#损失函数
def cost(w, b, x_train, y_train):
    total_cost=0
    m = len(x_train)
    for x_i, y_i in zip(x_train,y_train):
        total_cost += (y_i-w*x_i-b)**2
    return total_cost/m

#拟合函数
def fit(x_train, y_train):
    num = 0
    d = 0
    m = len(x_train)
    for x_i, y_i in zip(x_train, y_train):
        num += (x_i-x_ave)*(y_i-y_ave)
        d += (x_i-x_ave)**2
    a = num/d
    b = y_ave-a*x_ave
    return a, b


a, b = fit(x_train, y_train)
cost = cost(a, b, x_train, y_train)

print(a, b, cost)


plt.scatter(x_train,y_train)

pred_y= a*x_train+b

plt.plot(x_train, pred_y, c='r')
plt.show()
