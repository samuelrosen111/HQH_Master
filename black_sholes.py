import math
import time
import numpy as np
import random
import matplotlib.pyplot as plt


######################################################################## 💹 1 Basic Functions 💹 

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
        
    return 1-cumulative

def gbm(S0, mu, sigma, T, N):
    """
    Generate a Geometric Brownian Motion process.
    
    Parameters:
    S0 (float): initial stock price
    mu (float): expected return
    sigma (float): standard deviation of returns
    T (float): total time
    N (int): number of discrete time steps in the simulation
    
    Returns:
    np.array: simulated geometric Brownian motion process of one walk between time 0 and time 1.
    """
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

######################################################################## 💹 2 Black Scholes 💹 

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
 
######################################################################## 💹 3 Monte Carlo for Black Scholes 💹

def black_scholes_mc(S0, K, r, sigma, T, option_type, num_simulations, num_steps=252):
    """
    Estimates the price of a European call or put option using Monte Carlo simulation
    
    Parameters:
    S0 (float): the current stock price
    K (float): the strike price
    r (float): the risk-free interest rate
    sigma (float): the volatility of the underlying asset
    T (float): the time to maturity in years
    option_type (str): either "call" or "put" to specify the type of option
    num_simulations (int): the number of simulations to run
    num_steps (int): the number of time steps to use in the simulation
    
    Returns:
    float: the estimated price of the European call or put option
    """
    dt = T / (num_steps)
    S = np.zeros(num_simulations)
    S[::] = S0
    
    for i in range(1, num_steps):
        e = np.random.normal(size=num_simulations)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * e)
    
    if option_type == "call":
        payoff = np.maximum(S - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - S, 0)
    else:
        raise ValueError("Invalid option type, must be 'call' or 'put'")
    
    discount_factor = math.exp(-r * T)
    option_price = discount_factor * np.mean(payoff)
    return option_price

######################################################################## 💹 4 Heston 💹

def heston_Volatility(v0, theta, kappa, sigma, dt, num_steps):
    # Parameter:                        # Description and typical range

    # v0                                # Initial volatility (typical range: [0.01, 0.1])
    # theta                             # The mean to which the volatility reverts to. (typical range: [0.01, 0.1])
    # kappa                             # Mean reversion speed (typical range: [0.1, 10])
    # sigma                             # Volatility of volatility (typical range: [0.1, 0.5])
    # dt                                # Time step lenght (typical range: [0.001, 0.01])
    # num_steps                         # Number of steps (typical range: [10, 1000])

    """
    This function returns the volatility for the Hesotn model.
    Returned value is an array with volatility values
    """
    
    np.random.seed(None) #initialize random seed for the stochastic component - meaning we'll get new random numbers every iteration
    volatility = np.zeros(num_steps) # initialize the time array with the volatility for each timestep
    volatility[0] = v0 # first time step has the initial volatility

    for i in range(1, num_steps): 
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=1) # Random variable drawn from normal distribution with expectational value = 0 and std.dev = sqrt(dt)
        # Then updates the volatility with the heston formula (using Euler Maruyama for time-discretization)
        volatility[i] = volatility[i-1] + kappa * (theta - volatility[i-1]) * dt + sigma * np.sqrt(volatility[i-1]) * dW
    return volatility

def heston_model(S0, K, r, T, v0, theta, kappa, sigma, rho, dt, num_steps):
    # Parameters:
    # S0:         initial stock price                       # typical range: [0, 300] (usd)
    # K:          option strike price                       # typical range: [0, 300] (usd)
    # r:          risk-free interest rate                   # typical range: [0, 1] (percent)
    # T:          time to maturity                          # typical range: [0, 2] (years)
    # v0:         initial volatility                        # typical range: [0.01, 0.1]
    # theta:      mean reversion level of volatility        # typical range: [0.01, 0.1]
    # kappa:      mean reversion speed of volatility        # typical range: (0, 10)
    # sigma:      volatility of volatility                  # typical range: (0.1, 0.5)
    # rho:        correlation btw stock price and vol       # typical range: [-1, 1]
    # dt:         time step length                          # typical range: (T*0.01, T*0.000001) 
    # num_steps:  number of time steps                      # typical range: (100, 100000)
    # Calculate the stock price at each time step using the Heston model

    np.random.seed(None) # Initiates new random seed every run to have "new" random number
    stock_price = np.zeros(num_steps) # Array for stockprice valu
    stock_price[0] = S0 # initiates the starting price at t=0
    for i in range(1, num_steps):
        dZ1 = np.random.normal(loc=0, scale=np.sqrt(dt), size=1)
        dZ2 = rho * dZ1 + np.sqrt(1 - rho**2) * np.random.normal(loc=0, scale=np.sqrt(dt), size=1)
        v = heston_Volatility(v0, theta, kappa, sigma, dt, 1)[0]
        stock_price[i] = stock_price[i-1] * np.exp((r - 0.5*v) * dt + np.sqrt(v * dt) * dZ1)
        
    # Calculate the option price at time T
    option_price = np.maximum(stock_price[-1] - K, 0)
    option_price *= np.exp(-r * T)
    
    return option_price

def heston_simulation1(S0=100, K=100, r=0.05, T=1, v0=0.05, theta=0.05, kappa=1.5, sigma=0.3, rho=-0.5, dt=0.001, num_steps=1000):
    # Run the Heston model simulation
    np.random.seed(None)
    stock_price = np.zeros(num_steps)
    stock_price[0] = S0
    for i in range(1, num_steps):
        dZ1 = np.random.normal(loc=0, scale=np.sqrt(dt), size=1)
        dZ2 = rho * dZ1 + np.sqrt(1 - rho**2) * np.random.normal(loc=0, scale=np.sqrt(dt), size=1)
        v = heston_Volatility(v0, theta, kappa, sigma, dt, 1)[0]
        stock_price[i] = stock_price[i-1] * np.exp((r - 0.5*v) * dt + np.sqrt(v * dt) * dZ1)

    # Calculate the option price at time T
    option_price = np.maximum(stock_price[-1] - K, 0)
    option_price *= np.exp(-r * T)

    # Plot the stock price and volatility paths
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    ax[0].plot(np.arange(num_steps) * dt, stock_price)
    ax[0].set_ylabel('Stock price')
    ax[1].plot(np.arange(num_steps) * dt, heston_Volatility(v0, theta, kappa, sigma, dt, num_steps))
    ax[1].set_ylabel('Volatility')
    ax[1].set_xlabel('Time (years)')
    plt.show()

    return option_price


######################################################################## 💹 5 Heston REAL 💹


    # Set parameter values
    S0 = 100
    r = 0.05
    T = 1
    V0 = 0.1
    rho = -0.5
    kappa = 1.5
    theta = 0.1
    sigma = 0.1

    # Set strike price range
    K_range = np.linspace(80, 120, 41)

    # Calculate call option prices for each strike price
    call_prices = []
    for K in K_range:
        call_price = heston_price(S0, K, r, T, V0, rho, kappa, theta, sigma, "call")
        call_prices.append(call_price)

    # Plot the call option prices as a function of strike price
    plt.plot(K_range, call_prices)
    plt.xlabel("Strike Price")
    plt.ylabel("Call Option Price")
    plt.title("Heston Model Option Prices for Varying Strike Price")
    plt.show()

def heston():
    
    sigma = 0.2     # sigma (volatility of volatility): Typical range of values between 0.1 and 1.0
    theta = 0.04    # theta - long term price variance: Typical range of values between 0.01 and 0.5
    kappa = 1.5     # kappa - rate of reversion: Typical range of values between 0.1 and 5.0
    dW_st = np.random.normal(0, 1, 1000) # dW_st - brownian motion of asset price: Numpy array with simulated values
    dW_vt = np.random.normal(0, 1, 1000) # dW_vt - brownian motion of assets price variance: Numpy array with simulated values
    R = 0.5 # R - correlation between dW_st and dW_vt: Typical range of values between -1.0 and 1.0

    # Basic Heston model assumes that the price of an asset is determined by the stochastic process:

    # dS_t = mu * S0 * dt + np.sqrt(nu) * S0 * dWt
    # dS_t: Simulated change in stock price at time t
    # mu: Drift term. Represents the expected rate of return for the stock over a given time period. Typical range between 0.0 and 0.2.
    # S0: Initial stock price at time 0. Can be any stockprice value.
    # dt: Time increment. Specifies the time period over which the stock price change is simulated, typically between 0.0001 and 0.01 as a fraction of the whole time period.
    # nu: Volatility term. Represents the standard deviation of the stock's return over a given time period. Typical range between 0.05 and 0.5.
    # dWt: Simulated Brownian motion at time t used to model the stochastic (random) component of the stock price.
    # The Brownian motion represents the random fluctuations in the stock price that are due to factors such as market noise, unexpected news, and other unpredictable events.


    # Heston model's volatility equation:
    # dnu_t = kappa * (theta - nu_t) * dt + xi * np.sqrt(nu_t) * dWt_nu

    # dnu_t: Simulated change in stock price variance at time t
    # kappa: Rate of reversion. Represents the speed at which the variance reverts to its long-term mean. Typical range between 0.1 and 10.0.
    # theta: Long term variance. Represents the long-term average value of the stock price variance. Typical range between 0.0 and 1.0.
    # nu0: Initial variance value. Represents the starting value of the stock price variance. Typical range between 0.0 and 1.0.
    # dt: Time increment. Specifies the time period over which the stock price variance change is simulated, typically between 0.0001 and 0.01 as a fraction of the whole time period.
    # xi: Volatility of volatility. Represents the standard deviation of the volatility's return over a given time period. Typical range between 0.05 and 0.5.
    # dWt_nu: Simulated Brownian motion at time t used to model the stochastic (random) component of the stock price variance.
    # The Brownian motion represents the random fluctuations in the stock price variance that are due to factors such as market noise, unexpected news, and other unpredictable events.


def heston_volatility():
    dWt_nu = np.random.normal(0, np.sqrt(dt), size=1000)   # Simulated Brownian motion at time t
    nu_t = nu0
    dnu_t = 0
    for i in range(len(dWt_nu)):
        dnu_t = kappa * (theta - nu_t) * dt + xi * np.sqrt(nu_t) * dWt_nu[i]  # Simulated change in stock price variance at time t
        nu_t += dnu_t
    return {'dnu_t': dnu_t, 'kappa': kappa, 'theta': theta, 'nu0': nu0, 'dt': dt, 'xi': xi, 'dWt_nu': dWt_nu}





    
    

######################################################################## 💹 5 Heston 3.0 💹

def heston_price(S0, K, r, T, V0, rho, kappa, theta, sigma, option_type):
    """
    Calculates the price of a European call or put option using the Heston model

    Parameters:
    S0 (float): the current stock price
    K (float): the strike price
    r (float): the risk-free interest rate
    T (float): the time to maturity in years
    V0 (float): the initial volatility
    rho (float): the correlation between the Brownian motions
    kappa (float): the mean-reversion rate of volatility
    theta (float): the long-run mean of volatility
    sigma (float): the volatility of volatility
    option_type (str): either "call" or "put" to specify the type of option

    Returns:
    float: the price of the European call or put option
    """
    # Define the characteristic function of the Heston model
    def characteristic_function_Heston(u, S0, T, r, kappa, theta, V0, rho, sigma):
        a = kappa * theta
        b = kappa + rho * sigma
        c = sigma
        d = np.sqrt((b - rho * c * 1j * u)**2 - c**2 * (2 * u * 1j - u**2))
        x = np.log(S0)

        # Calculate g, G, C, and D
        g = (b - rho * c * 1j * u - d) / (b - rho * c * 1j * u + d)
        G = (1 - g * np.exp(-d * T)) / (1 - g)
        C = (r * u * 1j * T + a / c**2 * ((b - rho * c * 1j * u - d) * T - 2 * np.log(G)))
        D = (b - rho * c * 1j * u - d) / c**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

        # Return the characteristic function
        return np.exp(C + D * V0 + 1j * u * x)

    # Define the integrand for the P1 and P2 terms
    def integrand(u, S0, T, r, kappa, theta, V0, rho, sigma, K):
        return np.real(np.exp(-1j * u * np.log(K)) * characteristic_function_Heston(u - 1j, S0, T, r, kappa, theta, V0, rho, sigma) / (1j * u * characteristic_function_Heston(-1j, S0, T, r, kappa, theta, V0, rho, sigma)))

    # Calculate the P1 and P2 terms
    P1 = 0.5 + (1 / np.pi) * np.trapz([integrand(u, S0, T, r, kappa, theta, V0, rho, sigma, K) for u in np.arange(0.0001, 500, 0.1)])
    P2 = 0.5 + (1 / np.pi) * np.trapz([integrand(u, S0, T, r, kappa, theta, V0, rho, sigma, K) for u in np.arange(-500, -0.0001, 0.001)])
    
    # Calculate the call and put option prices
    int_value = S0 * P1 - K * np.exp(-r * T) * P2
    if option_type == "call":
        option_price = int_value + K * np.exp(-r * T) - S0
    elif option_type == "put":
        option_price = int_value + S0 - K * np.exp(-r * T)
    else:
        raise ValueError("Invalid option type, must be 'call' or 'put'")
        
    return option_price

def plot_heston_prices():
    # Define the parameter values
    S0 = 100
    K = 100
    r = 0.05
    T = 1
    V0 = 0.05**2
    rho = -0.7
    kappa = 4
    theta = 0.04**2
    sigma = 0.5

    # Generate a range of option prices to plot
    option_prices = np.linspace(80, 120, 2)

    # Calculate the Heston prices for each option price in the range
    call_prices = [heston_price(S0, K, r, T, V0, rho, kappa, theta, sigma, 'call') for K in option_prices]
    put_prices = [heston_price(S0, K, r, T, V0, rho, kappa, theta, sigma, 'put') for K in option_prices]

    # Plot the Heston call and put prices
    fig, ax = plt.subplots()
    ax.plot(option_prices, call_prices, label='Heston call')
    ax.plot(option_prices, put_prices, label='Heston put')
    ax.set_xlabel('Option price')
    ax.set_ylabel('Price')
    ax.set_title('Heston prices for European options')
    ax.legend()
    plt.show()


# Example usage of plot_heston_prices()



######################################################################## 🍭 ILLUSTRATION ('test functions') 🍭

# 1) Shows how the normal distribution works and the cumulative probability of it. Expectational value = 0, standard deviation = 1.
def illustrate_cumulative_prob():
    x = np.linspace(-3, 3, 1000)
    y_cumulative = []
    y_normal = []
    for i in x:
        y_cumulative.append(float(1-cumulative_prob(i)))
        y_normal.append(float((1 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * i * i)))
    fig, ax1 = plt.subplots()
    ax1.plot(x, y_cumulative, color='red')
    ax1.set_xlabel('X Value')
    ax1.set_ylabel('Cumulative Probability', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax2 = ax1.twinx()
    ax2.plot(x, y_normal, color='blue')
    ax2.set_ylabel('Normal Distribution', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title("Cumulative Probability and Normal Distribution")
    plt.show()

# 2) Shows how a Geometric Brownian motion looks 
def illustrate_gbm():
    # Set initial stock price, expected return, volatility, time, and number of steps
    S0 = 100 # Initial stock price
    mu = 0.1 # Expected value
    sigma = 0.2 #standard deviation of returns
    T = 1 # Total time of simulation
    N = 252 # Number of discrete time-steps in the simulation

    print(f"Illustration of GBM with parameters: \nInitial stock price: {S0} \nExpected return: {mu} \nVolatility: {sigma} \nTotal time: {T} \nDiscrete time steps: {N}")

    # Simulate the GBM process
    ammount_simulations = int(input("Enter how many simulation you want to run: "))

    for _ in range(1, ammount_simulations+1):
        current_GBM = gbm(S0, mu, sigma, T, N)
        plt.plot(current_GBM)

    #plt.plot(A) ghjgjgv
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Geometric Brownian Motion Simulation(s)")
    plt.show()
    # Calculate the average of all simulations
    average = np.mean(np.array([gbm(S0, mu, sigma, T, N) for _ in range(ammount_simulations)]), axis=0)
    # Plot the average of all simulations
    plt.plot(average, label='Average')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Average of the Geometric Brownian Motion Simulations.")
    plt.show()
    simulation_numbers = [5, 10, 50, 100, 500, 1000]
    for ammount_simulations in simulation_numbers:
        # Calculate the average of all simulations
        average = np.mean(np.array([gbm(S0, mu, sigma, T, N) for _ in range(ammount_simulations)]), axis=0)
        
        # Plot the average of all simulations
        plt.plot(average, label=f'Average of {ammount_simulations} simulations.')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title("Average of Geometric Brownian Motion Simulations")
        plt.show()

    # Illustrate the effect of changing expected return

    mu_values = [0.05, 0.4, 0.8, 1]
    for mu in mu_values:
        average = np.mean(np.array([gbm(S0, mu, sigma, T, N) for _ in range(ammount_simulations)]), axis=0)
        plt.plot(average, label=f'Expected return: {mu}')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title("Effect of Changing Expected Return (mu)")
    print(f"\n \n Illustration of GBM (changing mu) with parameters: \nInitial stock price: {S0} \nExpected return: {mu} \nVolatility: {sigma} \nTotal time: {T} \nDiscrete time steps: {N}")
    plt.show()

    # Illustrate the effect of changing volatility
    sigma_values = [0, 2, 4, 8]
    for sigma in sigma_values:
        average = np.mean(np.array([gbm(S0, mu, sigma, T, N) for _ in range(ammount_simulations)]), axis=0)
        plt.plot(average, label=f'Volatility: {sigma}')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title("Effect of Changing Volatility (sigma)")
    print(f"\n \n Illustration of GBM (changing sigma) with parameters: \nInitial stock price: {S0} \nExpected return: {mu} \nVolatility: {sigma} \nTotal time: {T} \nDiscrete time steps: {N}")
    print("Higher volatility = lower expected value over time. Somewhere around 3 things get out of hand.")
    plt.show()

# 3) Let the use play around with the BS formula to see how it works
def example_Run_Black_Scholes():
    # Example 1
    S0 = 100 # initial (current) stock price
    K = 95 # strike price
    r = 0.05 # risk-free interest rate
    sigma = 0.2 # volatility of underlying asset
    T = 1 # time to maturity normalized to 1 (for ex 1 year)
    option_type = "call"
    price_call = black_scholes(S0, K, r, sigma, T, option_type)
    print(f"\nExample 1, for the given values: \nS0 = {S0} \nK = {K} \nr = {r} \nsigma = {sigma} \nT = {T}")
    print(f"The price of the European {option_type} option is: {price_call}")

    # Example 2: Calculate the price of a European put option
    S0 = 100 # initial (current) stock price
    K = 105 # strike price
    r = 0.05 # risk-free interest rate
    sigma = 0.2 # volatility of underlying asset
    T = 1 # time to maturity normalized to 1 (for ex 1 year)
    option_type = 'put'
    price_put = black_scholes(S0, K, r, sigma, T, option_type)
    print(f"\nExample 2, for the given values: \nS0 = {S0} \nK = {K} \nr = {r} \nsigma = {sigma} \nT = {T}")
    print(f"The price of the European {option_type} option is: {price_put}")

    coco = 1
    while(coco==1):
        # User input for custom values
        S0 = float(input("Enter the current stock price (S0): "))
        K = float(input("Enter the strike price (K): "))
        r = float(input("Enter the risk-free interest rate (r): "))
        sigma = float(input("Enter the volatility of the underlying asset (sigma): "))
        T = float(input("Enter the time to maturity (T): "))
        option_type = input("Enter the option type ('call' or 'put'): ").lower()

        # Calculate option price based on user input
        price = black_scholes(S0, K, r, sigma, T, option_type)

        # Print the option price
        print(f"\nFor the given values: \nS0 = {S0} \nK = {K} \nr = {r} \nsigma = {sigma} \nT = {T}")
        print(f"The price of the European {option_type} option is: {price}")
        coco = int(input("For another example enter 1 to quit enter 0: "))

# 4) Plots the BS values for different parametes to see how they affect the price
def plot_Black_Scholes(S0=100, r=0.05, sigma=0.2, T=1, option_type='call', num_points=100):
    K_list = np.linspace(80, 120, num_points)
    call_values = np.zeros(num_points)
    put_values = np.zeros(num_points)
    
    for i, K in enumerate(K_list):
        call_values[i] = black_scholes(S0, K, r, sigma, T, 'call')
        put_values[i] = black_scholes(S0, K, r, sigma, T, 'put')
    
    # Print the parameter values
    print("\n--> Plot showing Call and Put prices for different strike prices, given:")
    print("S0 (current stock price) = ", S0)
    print("K (strike price) = ", K)
    print("r (risk-free interest rate) = ", r)
    print("sigma (volatility of the underlying asset) = ", sigma)
    print("T (time to maturity in years) = ", T)
  
    print("Values of prices: ")
    print("Strike price:  Call Value:  Put Value: ")
    for i in range (1,100):
        print("{:<14} {:<12} {:<12}".format(round(K_list[i], 2), round(call_values[i], 2), round(put_values[i], 2)))
        #time.sleep(0.1)


    plt.plot(K_list, call_values, 'b', label='Call option')
    plt.plot(K_list, put_values, 'r', label='Put option')
    plt.legend()
    plt.xlabel('Strike price (K)')
    plt.ylabel('Option value')
    plt.title('Black-Scholes option values for a range of strike prices')
    plt.show()

    # Plot Black-Scholes values for different sigmas

    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    put_values_sigma = np.zeros((len(sigma_list), num_points))

    for j, sigma_val in enumerate(sigma_list):
        for i, K in enumerate(K_list):
            put_values_sigma[j, i] = black_scholes(S0, K, r, sigma_val, T, 'put')

        plt.plot(K_list, put_values_sigma[j, :], label=f'Put option, sigma={sigma_val}')
    
    print("\n\n... For the same given values, here is a plot of sigma values vs price --> ")

    plt.legend()
    plt.xlabel('Strike price (K)')
    plt.ylabel('Option value')
    plt.title('Black-Scholes put option values for a range of strike prices and volatility')
    plt.show()

# 5) Plots the error of the monte carlo implementation of BS as a function of the time-step length. Here Simulations are kept high to rule that error-source out.
def plot_Option_Error_vs_TimeStepLength():
    # Note: these example values are the same as in "plot_Option_Error_vs_Simulations():"
    S0 = 100
    K = 110
    r = 0.05
    sigma = 0.2
    T = 1

    option_type = "call"
    black_scholes_analyticalValue_reference = black_scholes(S0, K, r, sigma, T, option_type)
    print("Black Scholes analytical value = ", black_scholes_analyticalValue_reference)

    num_steps_list = [2**i for i in range(1, 11)]

    mc_dt_values_list = []
    for current_ammount_steps in num_steps_list:
        mc_dt_values_list.append(black_scholes_mc(S0, K, r, sigma, T, option_type, 100000, current_ammount_steps))
    error_for_each_dt_list = []

    for mc_dt_value_current in mc_dt_values_list:
        error_for_each_dt_list.append(abs(black_scholes_analyticalValue_reference - mc_dt_value_current))
    
    print("\nThe function plots the absolute error between the Monte Carlo simulated option price and the analytical Black-Scholes option price, as a function of the number of time steps used in the simulation.")
    print("\nThe following table shows the Monte Carlo option price, the number of time steps used in the simulation, and the absolute error between the Monte Carlo simulated price and the analytical Black-Scholes price:")
    print("\nMC Option Price\t\t\tNum Time Steps\t\tAbsolute Error")
    for i in range(len(num_steps_list)):
        print(mc_dt_values_list[i], "\t\t\t", num_steps_list[i], "\t\t\t", error_for_each_dt_list[i])

    plt.title('Error in the Monte Carlo Approximation relative time steps')
    plt.loglog(num_steps_list, error_for_each_dt_list, 'bo-', label='Absolute Error')
    plt.xlabel('Number of Time Steps')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.show()

# 6) Plots the error of the monte carlo implementation of BS as a function of the amount of simulation. Here time-step-length are kept short to rule that error-source out.
def plot_option_error_vs_simulations():
    # Note: these example values are the same as in "plot_Option_Error_vs_TimeStepLength():"
    S0 = 100                #Initial price
    K = 110                 #Strike Price
    r = 0.05                # Risk free interest rate
    sigma = 0.2             # Volatility
    T = 1                   # Total duration time (normalized to 1)
    option_type = "call"


    num_sims_list=[10, 50, 100, 500,  1000, 5000, 10000, 50000, 100000]

    # Calculate Black-Scholes option price
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    bs_price = 0
    if option_type == "call":
        bs_price = S0 * cumulative_prob(d1) - K * math.exp(-r * T) * cumulative_prob(d2)
    elif option_type == "put":
        bs_price = K * math.exp(-r * T) * cumulative_prob(-d2) - S0 * cumulative_prob(-d1)

    # Calculate Monte Carlo option prices for different num_sims values
    mc_prices = []
    for num_sims in num_sims_list:
        mc_price = black_scholes_mc(S0, K, r, sigma, T, option_type, num_sims, num_steps=1000)
        mc_prices.append(mc_price)

    # Calculate absolute error between Monte Carlo and Black-Scholes prices
    errors = np.abs(np.array(mc_prices) - bs_price)

    # Calculate standard deviation of Monte Carlo option prices
    mc_std = np.std(mc_prices)

    
    # Print MC value, number of simulations, and error
    print("Black-Scholes price: ", bs_price)
    print("MC value\tNum_sims\tError")
    for i in range(len(mc_prices)):
        print("{:.4f}\t\t{}\t\t{:.4f}".format(mc_prices[i], num_sims_list[i], errors[i]))


    # Plot error vs. num_sims on a log-log chart
    plt.loglog(num_sims_list, errors, 'bo-', label='Absolute Error')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.show()

    # Print standard deviation of Monte Carlo option prices
    print("Standard deviation of Monte Carlo option prices: ", mc_std)

# 7) Plots an example development of Heston volatility with example values (different for each run due to new random numbers every run)
def example_plot_heston_volatility():
    # Set parameter values
    v0 = 0.05                 # Initial volatility (typical range: [0.01, 0.1])
    theta = 0.05              # The mean to which the volatility reverts to. (typical range: [0.01, 0.1])
    kappa = 1.5               # Mean reversion speed (typical range: [0.1, 10])
    sigma = 0.3               # Volatility of volatility (typical range: [0.1, 0.5])
    dt = 0.01                 # Time step length (typical range: [0.001, 0.01])
    num_steps = int(1/dt)     # Number of steps, set such that 1/dt is an integer value. Normalize total time-span to 1 for consistency

    # Call the heston_Volatility function
    volatility_over_time = heston_Volatility(v0, theta, kappa, sigma, dt, num_steps)
    


    # Print the parameter values
    print("Example run of Heston volatility with parameter values: ")
    print("v0: ", v0)
    print("theta: ", theta)
    print("kappa: ", kappa)
    print("sigma: ", sigma)
    print("dt: ", dt)
    print("num_steps: ", num_steps)

    # Print the volatility for each time step
    print("Volatility for each time step: \n ")
    time = np.linspace(0, (num_steps-1)*dt, num_steps)
    for i in range(num_steps):
        print("{:.2f}\t{:.4f}".format(time[i], volatility_over_time[i]))

    # Plot the volatility over time
    time = np.linspace(0, (num_steps-1)*dt, num_steps)
    plt.plot(time, volatility_over_time)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title('Heston Model Volatility over Time')
    plt.show()

# 8) Provides the user with typical values for Heston volatility and alow them to try different combination to see how the parameters affect the output
def user_input_plot_heston_volatility(): #GPT
    # Get parameter values from user input

    print("Note: for experimental purposes you can put in any values (even ouside the normal range) to see how the different parameters affect the volatility \n")

    v0 = float(input("Enter the initial volatility (typical range: [0, 0.3]): \t")) # For reference, Microsoft stock has about 30% volatility
    theta = float(input("Enter the mean to which the volatility reverts to (typical range: [0.01, 0.1]): \t"))
    kappa = float(input("Enter the mean reversion speed (typical range: [0.1, 10]): \t\t"))
    sigma = float(input("Enter the volatility of volatility (typical range: [0.1, 0.5]): \t"))
    dt = float(input("Enter the time step length (typical range: [0.001, 0.01]): \t\t"))
    num_steps = int(input("Enter the number of steps (typical range: [10, 1000]): \t\t"))


    # Print the parameter values in terminal
    print("\n Parameter values for the Heston volatility model:")
    print("v0: ", v0)
    print("theta: ", theta)
    print("kappa: ", kappa)
    print("sigma: ", sigma)
    print("dt: ", dt)
    print("num_steps: ", num_steps)

    # Call the heston_Volatility function and create array with volatility for each discrete time-step
    volatility_over_time = heston_Volatility(v0, theta, kappa, sigma, dt, num_steps)

    # Print the volatility for each time step in terminal
    print("\nVolatility for each time step:")

    print("Time step\tVolatility")
    for i, vol in enumerate(volatility_over_time):
        print("{:.2f}\t{:.4f}".format(i*dt, vol))

    # Plot the volatility over time
    time = np.linspace(0, (num_steps-1)*dt, num_steps)
    plt.plot(time, volatility_over_time, color='green')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title('Heston Model Volatility over Time')
    plt.show()

######################################################################## 💹 Main function that let's the user test different functions 💹
    
def main():
    # This function asks the user to pick which part of the code they want to run. 
    # Each illustrate funciton calls on a function of intetest and displays graphically how it works.
    print("Enter what you want to do:")
    print("1: Illustrate Cumulative Probability of Normal Distribution")
    print("2: Illustrate Geometric Brownian Motion example runs")
    print("3: Example runs - Black Scholes")
    print("4: Plotting of Black Scholes")
    print("5: Plot error (as afunction of # of timesteps) comparing black_scholes_mc and black_scholes")
    print("6: Plot error (as afunction of # of simulations) comparing black_scholes_mc and black_scholes")
    print("7: Illustration (with example values) of Heston volatility")
    print("8: Choose your own input values for heaston volatility")

    which_test_to_do = int(input("Enter the number of the test you would like to run: "))
    if which_test_to_do == 1:
        illustrate_cumulative_prob()
    elif which_test_to_do == 2:
        illustrate_gbm()
    elif which_test_to_do == 3:
        example_Run_Black_Scholes()
    elif which_test_to_do == 4:
        plot_Black_Scholes()
    elif which_test_to_do ==5:
        plot_Option_Error_vs_TimeStepLength()
    elif which_test_to_do ==6:
        plot_option_error_vs_simulations()
    elif which_test_to_do ==7:
        example_plot_heston_volatility()
    elif which_test_to_do ==8:
        user_input_plot_heston_volatility()
    else:
        print("")

    print("\n\n The end of the main program.")

main()
