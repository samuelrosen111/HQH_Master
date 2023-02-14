import math
import time
import numpy as np

def cumulative_prob(x): #cumulative_probability_standard_normal_distribution mu= 0, sigma = 1
    """
    Computes the cumulative standard normal distribution
    
    Parameters:
    x (float): the value for which the cumulative standard normal distribution is calculated
    
    Returns:
    float: the cumulative standard normal distribution value for the input x
    """
    # Coefficients in rational approximations
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    r = 0.2316419
    
    k = 1.0 / (1.0 + r * abs(x))
    cumulative = (1.0 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * x * x) * (a1 * k + a2 * k * k + a3 * k**3 + a4 * k**4 + a5 * k**5)
    
    if x < 0:
        cumulative = 1.0 - cumulative
        
    return cumulative


input_int = int(input("Optional illustration of 'cumulative_probability_standard_normal_distribution'. (1 = YES, 0 = NO) : "))
if(input_int):
    for i in np.arange(-3.1, 3.1, 0.1):
        print("Probability that value is: ", round(i,0), " (or above) is: ", round(cumulative_probability_standard_normal_distribution(i), 2))

def black_scholes(S0, K, r, sigma, T, option_type):
    """
    Calculates the price of a European call or put option using the Black-Scholes formula
    
    Parameters:
    S0 (float): the current stock price
    K (float): the strike price
    r (float): the risk-free interest rate
    sigma (float): the volatility of the underlying asset
    T (float): the time to maturity in years
    option_type (str): either "call" or "put" to specify the type of option
    
    Returns:
    float: the price of the European call or put option
    """
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == "call":
        value = S0 * cumulative_prob(d1) - K * math.exp(-r * T) * cumulative_prob(d2)
    elif option_type == "put":
        value = K * math.exp(-r * T) * cumulative_prob(-d2) - S0 * cumulative_prob(-d1)
    else:
        raise ValueError("Invalid option type, must be 'call' or 'put'")
        
    return value

# Example 1
S0 = 100 # initial (current) stock price
K = 95 # strike price
r = 0.05 # risk-free interest rate
sigma = 0.2 # volatility of underlying asset
T = 1 # time to maturity normalized to 1 (for ex 1 year)
option_type = "call"
print(f"The price of the European {option_type} option is: {black_scholes(S0, K, r, sigma, T, option_type)}")

# Example 2: Calculate the price of a European put option
S0 = 100 # initial (current) stock price
K = 105 # strike price
r = 0.05 # risk-free interest rate
sigma = 0.2 # volatility of underlying asset
T = 1 # time to maturity normalized to 1 (for ex 1 year)
option_type = 'put'

price = black_scholes(S0, K, r, sigma, T, option_type)
print("The price of the European {option_type} option is:", price)