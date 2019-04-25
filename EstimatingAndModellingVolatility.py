import numpy as np
import math
from Appendix import RandN
import matplotlib.pyplot as plt

def Simulating_GARCH():
    print("Enter inital stock price (S0)")
    S = float(input())
    print("Enter initial volatility (sigma)")
    sigma = float(input())
    print("Enter risk-free rate (r)")
    r = float(input())
    print("Enter dividend yield (q)")
    q = float(input())
    print("Enter length of each time period (Delta t)")
    dt = float(input())
    print("Enter number of time periods (N)")
    n = int(input())
    print("Enter theta")
    theta = float(input())
    print("Enter kappa")
    kappa = float(input())
    print("Enter lambda")
    lambda1 = float(input())

    logS = math.log(S)
    sqrdt = math.sqrt(dt)
    a = kappa * theta
    b = (1 - kappa) * (1 - lambda1)
    c = (1 - kappa) * lambda1
    GARCH = np.zeros((n, 3))
    GARCH[0, 1] = S
    GARCH[0, 2] = sigma

    for i in range(1, n):
        GARCH[i, 0] = i * dt
        y = sigma * RandN()
        logS = logS + (r - q - 0.5 * sigma * sigma) * dt + sqrdt * y
        S = math.exp(logS)
        GARCH[i, 1] = S
        sigma = math.sqrt(a + b * y * y + c * sigma * sigma)
        GARCH[i, 2] = sigma

    plt.subplot(1, 2, 1)
    plt.plot(GARCH[0:, 0], GARCH[0:, 1])
    plt.grid(True)
    plt.title('Stock Price Price')
    plt.xlabel('Time')

    plt.subplot(1, 2, 2)
    plt.plot(GARCH[0:, 0], GARCH[0:, 2])
    plt.grid(True)
    plt.title('Volatility')
    plt.xlabel('Time')

    plt.show()


def Simulating_Stochastic_Volatility():
    print("Enter inital stock price (S0)")
    S = float(input())
    print("Enter initial volatility (sigma)")
    sigma = float(input())
    print("Enter risk-free rate (r)")
    r = float(input())
    print("Enter dividend yield (q)")
    q = float(input())
    print("Enter length of each time period (Delta t)")
    dt = float(input())
    print("Enter number of time periods (N)")
    n = int(input())
    print("Enter theta")
    theta = float(input())
    print("Enter kappa")
    kappa = float(input())
    print("Enter Gamma")
    gamma = float(input())
    print("Enter Rho")
    rho = float(input())

    logS = math.log(S)
    var = sigma * sigma
    sqrdt = math.sqrt(dt)
    sqrrho = math.sqrt(1 - rho * rho)

    GARCH = np.zeros((n, 3))
    GARCH[0, 1] = S
    GARCH[0, 2] = sigma

    for i in range(1, n):
        GARCH[i, 0] = i * dt
        z1 = RandN()
        logS = logS + (r - q - 0.5 * sigma * sigma) * dt + sigma * sqrdt * z1
        S = math.exp(logS)
        GARCH[i, 1] = S
        z2 = RandN()
        zStar = rho * z1 + sqrrho * z2
        var = max(0, var + kappa * (theta - var) * dt + gamma * sigma * sqrdt * zStar)
        sigma = math.sqrt(var)
        GARCH[i, 2] = sigma

    plt.subplot(1, 2, 1)
    plt.plot(GARCH[0:, 0], GARCH[0:, 1])
    plt.grid(True)
    plt.title('Stock Price Price')
    plt.xlabel('Time')

    plt.subplot(1, 2, 2)
    plt.plot(GARCH[0:, 0], GARCH[0:, 2])
    plt.grid(True)
    plt.title('Volatility')
    plt.xlabel('Time')

    plt.show()

