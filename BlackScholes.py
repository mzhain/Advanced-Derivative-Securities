import numpy as np
import math
from Appendix import RandN, cumulative_standard_normal


def Black_Scholes_Call(S, K, r, sigma, q, T):
    # inputs
    """ S = initial stock price
        K = strike price
        r = risk-free rate
        sigma = volatility
        q = dividend yield
        T = time to maturity"""
    if sigma ==0:
        return round(max(0, math.exp(-q * T) * S - math.exp(-r * T) * K), 4)
    else:
        d1 = (math.log(S/K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        n1 = cumulative_standard_normal(d1)
        n2 = cumulative_standard_normal(d2)
        return round(math.exp(-q * T) * S * n1 - math.exp(-r * T) * K * n2, 4)


def Black_Scholes_Put(S, K, r, sigma, q, T):
    # inputs
    """ S = initial stock price
        K = strike price
        r = risk-free rate
        sigma = volatility
        q = dividend yield
        T = time to maturity"""
    if sigma ==0:
        return round(max(0, math.exp(-r * T) * K - math.exp(-q * T) * S), 4)
    else:
        d1 = (math.log(S/K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        n1 = cumulative_standard_normal(-d1)
        n2 = cumulative_standard_normal(-d2)
        return round(math.exp(-r * T) * K * n2 - math.exp(-q * T) * S * n1, 4)


def Black_Scholes_Call_Delta(S, K, r, sigma, q, T):
    # inputs
    """ S = initial stock price
        K = strike price
        r = risk-free rate
        sigma = volatility
        q = dividend yield
        T = time to maturity"""
    d1 = (math.log(S/K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    n1 = cumulative_standard_normal(d1)
    n2 = cumulative_standard_normal(d2)
    return round(math.exp(-q * T) * n1, 4)


def Black_Scholes_Call_Gamma(S, K, r, sigma, q, T):
    # inputs
    """ S = initial stock price
        K = strike price
        r = risk-free rate
        sigma = volatility
        q = dividend yield
        T = time to maturity"""
    d1 = (math.log(S/K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    n1 = cumulative_standard_normal(d1)
    n2 = cumulative_standard_normal(d2)
    nd1 = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    return round(math.exp(-q * T) * nd1 / (S * sigma * math.sqrt(T)), 4)


def Black_Scholes_call_Implied_Vol(S, K, r, q, T, CallPrice):
    # inputs
    """S = initial stock price
       K = strike price
       r = risk-free rate
       q = dividend yield
       T = time to maturity
       CallPrice = call price"""
    if CallPrice < math.exp(-q * T) * S - math.exp(-r * T) * K:
        print("Option price violates the arbitrage bound.")
        return
    else:
        tol = 10 ** -6
        lower = 0
        flower = Black_Scholes_Call(S, K, r, lower, q, T) - CallPrice
        upper = 1
        fupper = Black_Scholes_Call(S, K, r, upper, q, T) - CallPrice
        while fupper < 0:
            upper = 2 * upper
            fupper = Black_Scholes_Call(S, K, r, upper, q, T) - CallPrice
        guess = 0.5 * lower + 0.5 * upper
        fguess = Black_Scholes_Call(S, K, r, guess, q, T) - CallPrice
        while upper - lower > tol:
            if fupper * fguess < 0:
                lower = guess
                flower = fguess
                guess = 0.5 * lower + 0.5 * upper
                fguess = Black_Scholes_Call(S, K, r, guess, q, T) - CallPrice
            else:
                upper = guess
                fupper = fguess
                guess = 0.5 * lower + 0.5 * upper
                fguess = Black_Scholes_Call(S, K, r, guess, q, T) - CallPrice
        return guess


def Simulated_Delta_Hedge_Profit(S0, K, r, sigma, q, T, mu, M, N, pct):
    # inputs
    """ S0 = initial stock price
        K = strike price
        r = risk-free rate
        sigma = volatility
        q = dividend yield
        T = time to maturity
        mu = expected rate of return
        N = number of time periods
        M = number of simulations
        pct = percentile to be returned"""
    dt = T / N
    sigSqrdt = sigma * math.sqrt(dt)
    drift = (mu - q - 0.5 * sigma * sigma) * dt
    comp = math.exp(r * dt)
    div = math.exp(q * dt) - 1
    logS0 = math.log(S0)
    call0 = Black_Scholes_Call(S0, K, r, sigma, q, T)
    delta0 = Black_Scholes_Call_Delta(S0, K, r, sigma, q, T)
    cash0 = call0 - delta0 * S0     # initial cash position
    profit = np.zeros(M)
    for i in range(0, M):
        logS = logS0
        cash = cash0
        S = S0
        delta = delta0
        for j in range(1, N-1):
            logS = logS + drift + sigSqrdt * RandN()
            newS = math.exp(logS)
            newDelta = Black_Scholes_Call_Delta(newS, K, r, sigma, q, T - j * dt)
            cash = comp * cash + delta * S * div - (newDelta - delta) * newS
            S = newS
            delta = newDelta
        logS = logS + drift + sigSqrdt * RandN()
        newS = math.exp(logS)
        hedgeValue = comp * cash + delta * S * div + delta * newS
        profit[i] = hedgeValue - max(newS - K, 0)

    for i in profit:
        print(i)

    return np.percentile(profit, pct)



print("hello")
print(Black_Scholes_Call(50, 40, 0.05, 0.3, 0.02, 2))
print(Black_Scholes_Put(50, 40, 0.05, 0.3, 0.02, 2))
print(Black_Scholes_Call_Delta(50, 40, 0.05, 0.3, 0.02, 2))
print(Black_Scholes_Call_Gamma(50, 40, 0.05, 0.3, 0.02, 2))
print(Black_Scholes_call_Implied_Vol(50, 40, 0.05, 0.02, 2, 20))
print(Simulated_Delta_Hedge_Profit(50, 40, 0.05, 0.3, 0.02, 2, 0.12, 1000, 20, 0.05))
