import numpy as np
import math
from Appendix import RandN, cumulative_standard_normal
from ExchangeAndForwards import Generic_Option
from BlackScholes import Black_Scholes_Call

def BiNormalProb(a, b, rho):
    x = [0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334]
    y = [0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604]
    a1 = a / math.sqrt(2 * (1 - math.pow(rho, 2)))
    b1 = b / math.sqrt(2 * (1 - math.pow(rho, 2)))
    if a <= 0 and b <= 0 and rho <= 0:
        sum = 0
        for i in range(0, 4):
            for j in range(0, 4):
                z1 = a1 * (2 * y[i] - a1)
                z2 = b1 * (2 * y[j] - b1)
                z3 = 2 * rho * (y[i] - a1) * (y[j] - b1)
                sum = sum + x[i] * x[j] * math.exp(z1 + z2 + z3)
        return sum * math.sqrt(1 - math.pow(rho, 2)) / math.pi
    elif a <= 0 and b >= 0 and rho >= 0:
        return cumulative_standard_normal(a) - BiNormalProb(a, b * -1, rho * -1)
    elif a >= 0 and b <= 0 and rho >= 0:
        return cumulative_standard_normal(b) - BiNormalProb(a * -1, b, rho * -1)
    elif a >= 0 and b >= 0 and rho <= 0:
        sum = cumulative_standard_normal(a) + cumulative_standard_normal(b)
        return sum - 1 + BiNormalProb(a * -1, b * -1, rho)
    elif a * b * rho > 0:
        rho1 = (rho * a - b) * np.sign(a) / np.sign(math.pow(a, 2) - 2 * rho * a * b + math.pow(b, 2))
        rho2 = (rho * b - a) * np.sign(b) / math.sqrt(math.pow(a, 2) - 2 * rho * a * b + math.pow(b, 2))
        Delta = (1 - np.sign(a) * np.sign(b)) / 4
        return BiNormalProb(a, 0, rho1) + BiNormalProb(b, 0, rho2) - Delta


def Forward_Start_Call(S, r, sigma, q, Tset, TCall):
    # inputs
    """S = initial stock price
       r = risk-free rate
       sigma = volatility
       q = dividend yield
       Tset = time until the strike is set
       TCall = time until call matures >= Tset"""
    P1 = math.exp(-q * TCall) * S
    P2 = math.exp(-q * Tset - r * (TCall - Tset)) * S
    return Generic_Option(P1, P2, sigma, TCall - Tset)


def Call_On_Call(S, Kc, Ku, r, sigma, q, Tc, Tu):
    # inputs
    """S = initial stock price
       Kc = strike price of compound call
       Ku = strike price of underlying call option
       r = risk-free rate
       sigma = volatility
       q = dividend yield
       Tc = time to maturity of compound call
       Tu = time to maturity of underlying call >= Tc"""
    tol = math.pow(10, -6)
    lower = 0
    upper = math.exp(q * (Tu - Tc)) * (Kc + Ku)
    guess = 0.5 * lower + 0.5 * upper
    flower = -Kc
    fupper = Black_Scholes_Call(upper, Ku, r, sigma, q, Tu - Tc) - Kc
    fguess = Black_Scholes_Call(guess, Ku, r, sigma, q, Tu - Tc) - Kc
    while upper - lower > tol:
        if fupper * fguess < 0:
            lower = guess
            flower = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = Black_Scholes_Call(guess, Ku, r, sigma, q, Tu - Tc) - Kc
        else:
            upper = guess
            fupper = fguess
            guess = 0.5 * lower + 0.5 * upper
            fguess = Black_Scholes_Call(guess, Ku, r, sigma, q, Tu - Tc) - Kc
    Sstar = guess
    #print(Sstar)

    d1 = (math.log(S / Sstar) + (r - q + math.pow(sigma, 2) / 2) * Tc) / (sigma * math.sqrt(Tc))
    d2 = d1 - sigma * math.sqrt(Tc)
    d1prime = (math.log(S / Ku) + (r - q + math.pow(sigma, 2) / 2) * Tu) / (sigma * math.sqrt(Tu))
    d2prime = d1prime - sigma * math.sqrt(Tu)
    rho = math.sqrt(Tc / Tu)
    #print(d1)
    #print(d2)
    #print(rho)
    N2 = cumulative_standard_normal(d2)
    print(N2)
    M1 = BiNormalProb(d1, d1prime, rho)
    print(M1)
    M2 = BiNormalProb(d2, d2prime, rho)
    print(M2)
    return -math.exp(-r * Tc) * Kc * N2 + math.exp(-q * Tu) * S * M1 - math.exp(-r * Tu) * Ku * M2





print(BiNormalProb(0, 1, 0.3))
print(Forward_Start_Call(50, 0.05, 0.3, 0.02, 1, 2))
print(Call_On_Call(50, 10, 40, 0.05, 0.3, 0.02, 1, 2)) # not working
