import numpy as np
import math
from Appendix import RandN, cumulative_standard_normal
from ExchangeAndForwards import Generic_Option

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


print(BiNormalProb(0, 1, 0.5))
print(Forward_Start_Call(50, 0.05, 0.3, 0.02, 1, 2))
