import numpy as np
import math
from Appendix import RandN, cumulative_standard_normal

def Generic_Option(P1, P2, sigma, T):
    # inputs
    """P1 = present value of asset to be received
       P2 = present value of asset to be delivered
       sigma = volatility
       T = time to maturity"""
    x = (math.log(P1/ P2) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    y = x - sigma * math.sqrt(T)
    N1 = cumulative_standard_normal(x)
    N2 = cumulative_standard_normal(y)
    return P1 * N1 - P2 * N2


def Margrabe(S1, S2, sigma, q1, q2, T):
    # inputs
    """S1 = price of asset to be received
       S2 = price of asset to be delivered
       sigma = volatility of ratio of prices
       q1 = dividend yield of asset to be received
       q2 = dividend yield of asset to be delivered
       T = time to maturity"""
    return Generic_Option(math.exp(-q1 * T) * S1, math.exp(-q2 * T) * S2, sigma, T)


def Black_Call(F, K, P, sigma, T):
    # inputs
    """F = forward price
       K = strike price
       P = price of discount bond maturing when forward matures
       sigma = volatility of forward price
       T = time to maturity"""
    return Generic_Option(P * F, P * K, sigma, T)


def Black_Put(F, K, P, sigma, T):
    # inputs
    """F = forward price
       K = strike price
       P = price of discount bond maturing when forward matures
       sigma = volatility of forward price
       T = time to maturity"""
    return Generic_Option(P * K, P * F, sigma, T)


def Margrabe_Deferred(S1, S2, sigma, q1, q2, Tmat, Texch):
    # inputs
    """S1 = price of asset to be received
       S2 = price of asset to be delivered
       sigma = volatility of ratio of prices
       q1 = dividend yield of asset to be received
       q2 = dividend yield of asset to be delivered
       Tmat = time to maturity of option
       Texch = time until exchange >= TOption"""
    return Generic_Option(math.exp(-q1 * Texch) * S1, math.exp(-q2 * Texch) * S2, sigma, Tmat)

print(Generic_Option(50, 40, 0.3, 2))
print(Margrabe(50, 40, 0.3, 0.02, 0.01, 2))
print(Black_Call(50, 40, 0.9, 0.3, 2))
print(Black_Put(50, 40, 0.9, 0.3, 2))
print(Margrabe_Deferred(50, 40, 0.3, 0.02, 0.01, 2, 3))
