"""
-----------------------
# GradientDescent
# Created by ysbaekFox
# Date : 2021.03.28
# brief :
-----------------------
# Reference page
 - https://teddylee777.github.io/scikit-learn/gradient-descent
 - https://angeloyeo.github.io/2020/08/16/gradient_descent.html
 - loss function : https://kolikim.tistory.com/36
"""

import numpy as np
import matplotlib.pyplot as plt


def make_linear(w=0.5, b=0.8, size=50, noise=1.0):
    x = np.random.rand(size)
    y = w * x + b
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color='r', label=f'y = {w}*x {b}')
    plt.scatter(x, yy, label='data')
    plt.legend(fontsize=20)
    plt.show()
    print(f'w: {w}, b: {b}')
    return x, yy


def GradientDesecent(epoch, eta, x, y):
    erros = []
    w = np.random.uniform(low=-1.0, high=1.0)
    b = np.random.uniform(low=-1.0, high=1.0)

    for idx in range(epoch):
        y_hat = x * w + b
        error = ((y_hat - y)**2).mean() # 평균 제곱 오
        if error < 0.0005:
            break

        w += -1 * eta * ((y_hat - y) * x).mean()
        b += -1 * eta * (y_hat - y).mean()

        erros.append(error)

        if 0 == idx % 5:
            print("{0:2} w = {1:.5f}, b = {2:.5f}, error = {3:.5f},".format(epoch, w, b, error))

    return erros


if __name__ == '__main__':
    x, y = make_linear(w=0.3, b=0.5, size=100, noise=0.01)
    errors = GradientDesecent(epoch=500, eta=0.1, x=x, y=y)

    plt.figure(figsize=(10, 7))
    plt.plot(errors)
    plt.show()
