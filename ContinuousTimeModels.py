import numpy as np
import math
from Appendix import RandN
import matplotlib.pyplot as plt

def Simulating_Brownian_Motion():
    print("Enter the length of the time period (T)")
    T = float(input())
    print("Enter the number of periods (N)")
    N = int(input())

    dt = T / N
    sqrdt = math.sqrt(dt)
    BM = np.zeros((N, 2))

    for i in range(1, N):   # Simulating Brownian Motion
        BM[i, 0] = i * dt
        BM[i, 1] = BM[i-1, 1] + sqrdt * RandN()

    plt.plot(BM[0:, 0], BM[0:, 1])
    plt.grid(True)
    plt.title('Brownian Motion')
    plt.ylabel('B(T)')
    plt.xlabel('Time')
    plt.show()
    return BM

def Simulating_Geometric_Brownian_Motion():
    print("Enter the length of the time period (T)")
    T = float(input())
    print("Enter the number of periods (N)")
    N = int(input())
    print("Enter the initial stock price (S)")
    S = float(input())
    print("Enter the expected rate of return (mu)")
    mu = float(input())
    print("Enter the volatility (sigma)")
    sigma = float(input())

    dt = T / N
    sigSqrdt = sigma * math.sqrt(dt)
    drift = (mu - 0.5 * sigma * sigma) * dt
    GBM = np.zeros((N, 2))
    GBM[0, 1] = S

    for i in range(1, N):
        GBM[i, 0] = i * dt
        GBM[i, 1] = math.exp(math.log(GBM[i-1, 1]) + drift + sigSqrdt * RandN())

    plt.plot(GBM[0:, 0], GBM[0:, 1])
    plt.grid(True)
    plt.title('Strock Price')
    plt.ylabel('S(T)')
    plt.xlabel('Time')
    plt.show()
    return GBM





