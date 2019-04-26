import numpy as np
import math
from Appendix import RandN, cumulative_standard_normal

def European_Call_MC(S, K, r, sigma, q, T, M):
    # inputs
    """S = initial stock price
       K = strike price
       r = risk-free rate
       sigma = volatility
       q = dividend yield
       T = time to maturity
       M = number of simulations"""
    logS0 = math.log(S)
    drift = (r - q - 0.5 * sigma * sigma) * T
    sigSqrt = sigma * math.sqrt(T)
    UpChange = math.log(1.01)
    DownChange = math.log(0.99)
    SumCall = 0
    SumCallChange = 0
    SumPathwise = 0
    for i in range(1, M):
        logS = logS0 + drift + sigSqrt * RandN()
        callV = max(0, math.exp(logS) - K)
        SumCall = SumCall + callV
        logSu = logS + UpChange
        CallVu = max(0, math.exp(logSu) - K)
        logSd = logS + DownChange
        CallVd = max(0, math.exp(logSd) - K)
        SumCallChange = SumCallChange + CallVu - CallVd
        if math.exp(logS) > K:
            SumPathwise = SumPathwise + math.exp(logS) / S
    CallV = math.exp(-r * T) * SumCall / M
    Delta1 = math.exp(-r * T) * SumCallChange / (M * 0.02 * S)
    Delta2 = math.exp(-r * T) * SumPathwise / M
    results = [CallV, Delta1, Delta2]
    print(results)
    return results


def  Eur_Call_GARCH_MC(S, K, r, sigma0, q, T, N, kappa, theta, lambda1, M):
    # inputs
    """S = initial stock price
       K = strike price
       r = risk-free rate
       sigma0 = initial volatility
       q = dividend yield
       T = time to maturity
       N = number of time periods
       kappa = GARCH parameter
       theta = GARCH parameter
       lambda = GARCH parameter
       M = number of simulations"""
    dt = T / N
    sqrdt = math.sqrt(dt)
    a = kappa * theta
    b = (1- kappa) * (1 - lambda1)
    c = (1 - kappa) * (1 - lambda1)
    logS0 = math.log(S)
    SumCall = 0
    SumCallSq = 0
    for i in range(1, M):
        logS = logS0
        sigma = sigma0
        for j in range(1, N):
            y = sigma * RandN()
            logS = logS + (r - q - 0.5 * sigma * sigma) * dt + sqrdt * y
            sigma = math.sqrt(a + b * y * y + c * sigma * sigma)
        CallV = max(0, math.exp(logS) - K)
        SumCall = SumCall + CallV
        SumCallSq = SumCallSq + CallV * CallV
    CallV = math.exp(-r * T) * SumCall / M
    StdError = math.exp(-r * T) * math.sqrt((SumCallSq - SumCall * SumCall / M) / (M * (M - 1)))
    results = [CallV, StdError]
    print(results)
    return results


def European_Call_Binomial(S, K, r, sigma, q, T, N):
    # inputs
    """S = initial stock price
       K = strike price
       r = risk-free rate
       sigma = volatility
       q = dividend yield
       T = time to maturity
       N = number of time periods"""
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1/ u
    pu = (math.exp((r - q) * dt) - d) / (u - d)
    pd = 1 - pu
    u2 = u * u
    S = S * math.pow(d, N)
    prob = math.pow(pd, N)
    CallV = prob * max(S - K, 0)
    for i in range(1, N):
        S = S * u2
        prob = prob * (pu / pd) * (N - i + 1) / i
        CallV = CallV + prob * max(S - K, 0)
    results = math.exp(-r * T) * CallV
    print(results)
    return results


def American_Put_Binomial(S0, K, r, sigma, q, T, N):
    # inputs
    """S0 = initial stock price
       K = strike price
       r = risk-free rate
       sigma = volatility
       q = dividend yield
       T = time to maturity
       N = number of time periods"""
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    pu = (math.exp((r - q) * dt) - d) / (u - d)
    dpu = math.exp(-r * dt) * pu
    dpd = math.exp(-r * dt) * (1 - pu)
    u2 = u * u
    S = S0 * math.pow(d, N)
    PutV = np.zeros((N))
    PutV[0] = max(K - S, 0)
    for i in range(1, N):
        S = S * u2
        PutV[i] = max(K - S, 0)
    for j in range(N - 1, 0, -1):
        S = S0 * math.pow(d, j)
        PutV[0] = max(K - S, dpd * PutV[0] + dpu * PutV[1])
        for k in range(1, j):
            S = S * u2
            PutV[k] = max(K - S, dpd * PutV[k] + dpu * PutV[k + 1])
    results = PutV[0]
    print(results)
    return results


def American_Put_Binomial_DG(S0, K, r, sigma, q, T, N):
    # inputs
    """S0 = initial stock price
       K = strike price
       r = risk-free rate
       sigma = volatility
       q = dividend yield
       T = time to maturity
       N = number of time periods"""
    dt = T / N
    newN = N + 2
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    pu = (math.exp((r - q) * dt) - d) / (u - d)
    dpu = math.exp(-r * dt) * pu
    dpd = math.exp(-r * dt) * (1 - pu)
    u2 = u * u
    S = S0 * math.pow(d, newN)
    PutV = np.zeros((N + 2))
    PutV[0] = max(K - S, 0)
    for j in range(1, newN):
        S = S * u2
        PutV[j] = max(K - S, 0)
    for i in range(newN - 1, 2, -1):
        S = S0 * math.pow(d, i)
        PutV[0] = max(K - S, dpd * PutV[0] + dpu * PutV[1])
        for j in range(1, i):
            S = S * u2
            PutV[j] = max(K - S, dpd * PutV[j] + dpu * PutV[j + 1])
    Su = S0 * u2
    Sd = S0 / u2
    DeltaU = (PutV[2] - PutV[1]) / (Su - S0)
    DeltaD = (PutV[1] - PutV[0]) / (S0 - Sd)
    distance = S0 * (u2 - d * d)
    Delta = (PutV[2] - PutV[0]) / distance
    Gamma = 2 * (DeltaU - DeltaD) / distance
    results = [PutV[1], Delta, Gamma]
    print(results)
    return results



European_Call_MC(50, 40, 0.05, 0.3, 0.02, 2, 1000)
Eur_Call_GARCH_MC(50, 40, 0.05, 0.3, 0.02, 2, 24, 0.5, 0.09, 0.5, 1000)
European_Call_Binomial(50, 40, 0.05, 0.3, 0.02, 2, 50)
American_Put_Binomial(50, 40, 0.05, 0.3, 0.02, 2, 50)
American_Put_Binomial_DG(50, 40, 0.05, 0.3, 0.02, 2, 50)