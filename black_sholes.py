import math
import time
import numpy as np
import matplotlib.pyplot as plt


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

def illustrate_black_scholes():
    # Example 1
    S0 = 100 # initial (current) stock price
    K = 95 # strike price
    r = 0.05 # risk-free interest rate
    sigma = 0.2 # volatility of underlying asset
    T = 1 # time to maturity normalized to 1 (for ex 1 year)
    option_type = "call"
    price_call = black_scholes(S0, K, r, sigma, T, option_type)
    print(f"\nExample 1, for the given values: \nS0 = {S0} \nK = {K} \nr = {r} \nsigma = {sigma} \nT = {T}")
    print(f"\nThe price of the European {option_type} option is: {price_call}")

    # Example 2: Calculate the price of a European put option
    S0 = 100 # initial (current) stock price
    K = 105 # strike price
    r = 0.05 # risk-free interest rate
    sigma = 0.2 # volatility of underlying asset
    T = 1 # time to maturity normalized to 1 (for ex 1 year)
    option_type = 'put'
    price_put = black_scholes(S0, K, r, sigma, T, option_type)
    print(f"The price of the European {option_type} option is:", price_put)

    time.sleep(10)

    #New input values for the graph of put and call options

    S0 = 50 # initial (current) stock price
    K_call = 55 # strike price for call option
    K_put = 30 # strike price for put option
    r = 0.05 # risk-free interest rate
    sigma = 0.2 # volatility of underlying asset

    #time values to calculate the prices
    time_values = np.arange(0.0, 10, 0.1)
    call_prices = []
    put_prices = []

    #calculating the prices for both call and put options
    for T in time_values:
        call_price = -black_scholes(S0, K_call, r, sigma, T, "call")
        call_prices.append(call_price)
        put_price = -black_scholes(S0, K_put, r, sigma, T, "put")
        put_prices.append(put_price)

    plt.plot(time_values, call_prices, label="European call option")
    plt.plot(time_values, put_prices, label="European put option")
    plt.xlabel("Time to maturity (in years)")
    plt.ylabel("Option price")
    plt.legend()
    plt.show()

    # Printing the prices for put options at different time to maturity
    if(1):
        for t, put_price in zip(time_values, put_prices):
            print(f"Time T (in years) = {round(t,1)} Put price at this time: {round(put_price, 1)}")

illustrate_black_scholes()

def main():
    print("Enter what you want to do:")
    print("1. Illustrate Cumulative Probability of Normal Distribution")
    print("2. Illustrate Geometric Brownian Motion example runs")
    print("3 Illustrate Black Scholes")
    print("Any other number to skip")

    test = int(input("Enter the number of the test you would like to run: "))
    if test == 1:
        illustrate_cumulative_prob()
    elif test == 2:
        illustrate_gbm()
    elif test == 3:
        illustrate_black_scholes()
    else:
        print("")

    print("The end.")

def simulate_options(S0, K_call, K_put, r, sigma, T, N, num_simulations):
    """
    Simulate European call and put option prices using a geometric brownian motion process.
    
    Parameters:
    S0 (float): initial stock price
    K_call (float): strike price for call option
    K_put (float): strike price for put option
    r (float): risk-free interest rate
    sigma (float): standard deviation of returns
    T (float): total time
    N (int): number of discrete time steps in the simulation
    num_simulations (int): number of simulation walks to average over
    
    Returns:
    tuple: mean call option price and mean put option price, both over num_simulations walks
    """
    T = float(T)
    dt = T / N
    call_prices = np.zeros(num_simulations)
    put_prices = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        S = gbm(S0, r, sigma, T, N)
        for j, t in enumerate(np.linspace(0, T, N)):
            call_price = black_scholes(S[j], K_call, r, sigma, T - t, "call")
            call_prices[i] += call_price * np.exp(-r * (T - t - j * dt)) * dt
            put_price = black_scholes(S[j], K_put, r, sigma, T - t, "put")
            put_prices[i] += put_price * np.exp(-r * (T - t - j * dt)) * dt
            
    mean_call_price = call_prices.mean()
    mean_put_price = put_prices.mean()
    
    return mean_call_price, mean_put_price



#main()




"""""
# Example input values
S0 = 50
K_call = 55
K_put = 45
r = 0.05
sigma = 0.2
T = 1
option_type = "call"
N = 1000
num_simulations = 1000

# Call the simulate_options function with the example input values
option_prices = simulate_options(S0, K_call, K_put, r, sigma, T, N, num_simulations)


# Plot a histogram of the simulated option prices
plt.hist(option_prices, bins=50, edgecolor='black')
plt.xlabel("Option price")
plt.ylabel("Frequency")
plt.title(f"Histogram of simulated {option_type} option prices")
plt.show()
"""