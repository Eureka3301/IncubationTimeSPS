import numpy as np

import matplotlib.pyplot as plt

def model(X, p1, p2):
    X0 = p1/p2
    if X < X0:
        return p1 + p2 * X
    else:
        return 2 * np.sqrt(p1 * p2 * X)

def plot_model(p1, p2):
    xx = np.linspace(1e-9, 1e-3, 100)
    yy = np.array([model(xx[i], p1, p2) for i in range(100)])

    plt.plot(xx, yy)

    plt.show()

def gen_data(sigcr, tau, E):
    
    xx = np.linspace(1e+1, 1e+4, 10)
    yy = np.array([model(xx[i], sigcr, tau, E) for i in range(len(xx))])
    
    return (xx, yy)


def test():

    E = 1
    sigcr = 100
    tau = 10

    xx0, yy0 = gen_data(sigcr, tau, E)

    plt.plot(xx0, yy0)

    plt.show()


def LSM(X, Y, model):
    pass