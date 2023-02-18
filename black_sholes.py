import math
import time
import numpy as np
import matplotlib.pyplot as plt



# Functions being used:



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



#Functions to demostarte the code and run examples:



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

def example_run_black_scholes():
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

def plot_black_scholes(S0=100, r=0.05, sigma=0.2, T=1, option_type='call', num_points=100):
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

    plt.plot(K_list, call_values, 'b', label='Call option')
    plt.plot(K_list, put_values, 'r', label='Put option')
    plt.legend()
    plt.xlabel('Strike price (K)')
    plt.ylabel('Option value')
    plt.title('Black-Scholes option values for a range of strike prices')
    plt.show()

    #plots sigma also

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

# Main function to let the use navigate the code


def main():
    # This function asks the user to pick which part of the code they want to run. 
    # Each illustrate funciton calls on a function of intetest and displays graphically how it works.
    print("Enter what you want to do:")
    print("1: Illustrate Cumulative Probability of Normal Distribution")
    print("2: Illustrate Geometric Brownian Motion example runs")
    print("3: Example runs - Black Scholes")
    print("4: Plotting of Black Scholes")

    test = int(input("Enter the number of the test you would like to run: "))
    if test == 1:
        illustrate_cumulative_prob()
    elif test == 2:
        illustrate_gbm()
    elif test == 3:
        example_run_black_scholes()
    elif test == 4:
        plot_black_scholes()
    else:
        print("")

    print("\n\n The end of the main program.")

main()
