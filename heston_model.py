# Kolla upp referens mot annan kod och sätt samma värden se om samma resultat
# Jämför med heston biliotek
# Ett referensvärde
# Maxad simulering med 10000000 slumptal 1000 tidsteg = referensvärde
# Simuleringar upp till 200,000 (med 1000, 10.000 ish) några punkter ()
# Kolla mot tidssteg och si uleringar som med BS (maxa alla parametrar som referens)

# sqrt(num_sim) = O(1/2) convergence. 
# Euler Maruyama convergence 1. EF fast för SDE.
# EF missar om stort tidssteg + stor derivata


# Next: HQH (läs på heldag)
# MC istället för COS



# 1) Verifiera Heston
# 2) Kolla konvergens för Heston
# 3) 

import numpy as np
import matplotlib.pyplot as plt

def heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r):
    """
    (Credit for correct theory: https://quantpy.com.au/stochastic-volatility-models/simulating-heston-model-in-python/)
    Simulate asset prices and variance using the Heston model.
    
    Parameters:
    - S0: initial asset price (depends on asset)
    - v0: initial variance (typical range: 0 < v0 < 1)
    - rho: correlation between asset returns and variance (typical range: -1 < rho < 1)
    - kappa: rate of mean reversion in variance process (typical range: 0 < kappa < 10)
    - theta: long-term mean of variance process (typical range: 0 < theta < 1)
    - sigma: volatility of volatility or the degree of randomness in the variance process (typical range: 0 < sigma < 1)
    - T: time of simulation in years (typical range: 0 < T < 10)
    - num_steps: number of time steps (typical range: 10 < N < 1000)
    - num_sims: number of scenarios/simulations (typical range: 10 < M < 1000)
    - r: risk-free interest rate (typical range: 0 < r < 0.1) ... 0.1 would be 10%

    
    Returns:
    - numpy array of asset prices over time (shape: (N+1, M))
    - numpy array of variances over time (shape: (N+1, M))
    """

    # Calculate time increment
    dt = T/num_steps
    
    # Set initial drift and covariance matrix
    drift_term = [0,0]
    covariance_matrix = np.array([[1,rho], [rho,1]])
    
    # Create arrays to store asset prices and variances over time
    stonk = np.full(shape=(num_steps+1,num_sims), fill_value=S0)
    volatility = np.full(shape=(num_steps+1,num_sims), fill_value=v0)
    
    # Sample correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(drift_term, covariance_matrix, (num_steps,num_sims))
    
    # Calculate asset prices and variances over time
    for i in range(1,num_steps+1):
        stonk[i] = stonk[i-1] * np.exp( (r - 0.5*volatility[i-1])*dt + np.sqrt(volatility[i-1] * dt) * Z[i-1,:,0] )
        volatility[i] = np.maximum(volatility[i-1] + kappa*(theta-volatility[i-1])*dt + sigma*np.sqrt(volatility[i-1]*dt)*Z[i-1,:,1],0)

    return stonk, volatility

def illustrate_heston():
    """
    Illustrates the Heston model by simulating asset prices and variances over time.
    """

    # Simulation Parameters
    S0 = 100.0             # initial asset price (typical range: varies by asset)
    T = 1.0                # time horizon in years (typical range: 0 < T < 10)
    r = 0.02               # risk-free interest rate (typical range: 0 < r < 0.1)
    num_steps = 252        # number of time steps in simulation (typical range: 10 < num_steps < 1000)

    # Heston Parameters
    kappa = 3              # rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10)
    theta = round(0.20**2, 2)        # long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1)
    v0 = 0.25**2           # initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1)
    rho = 0.7              # correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1)
    sigma = 0.6            # volatility of volatility (typical range: 0 < sigma < 1)

    # Print function purpose and parameter values
    print("\n\n Running the illustrate_heston function to simulate asset prices and variances over time.\n")
    print("Given example variable values:\n")
    print(f"S0\t\t{S0}\t\tInitial asset price (typical range: varies by asset)")
    print(f"T\t\t{T}\t\tTime horizon in years (typical range: 0 < T < 10)")
    print(f"r\t\t{r}\t\tRisk-free interest rate (typical range: 0 < r < 0.1)")
    print(f"num_steps\t{num_steps}\t\tNumber of time steps in simulation (typical range: 10 < num_steps < 1000)")
    print(f"kappa\t\t{kappa}\t\tRate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10)")
    print(f"theta\t\t{theta:.2f}\t\tLong-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1)")
    print(f"v0\t\t{v0}\t\tInitial variance under risk-neutral dynamics (typical range: 0 < v0 < 1)")
    print(f"rho\t\t{rho}\t\tCorrelation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1)")
    print(f"sigma\t\t{sigma}\t\tVolatility of volatility (typical range: 0 < sigma < 1)")



    # Prompt user for the number of simulations they want to run
    num_sims = int(input("Enter the number of simulations to run: "))

    # simulate asset prices and variances using the Heston model
    simulated_stock_price, simulated_volatility = heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, num_steps, num_sims, r)
    
    # plot asset prices and variances over time
    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12,5))
    time = np.linspace(0,T,num_steps+1)
    ax1.plot(time,simulated_stock_price)
    ax1.set_title('Heston Model Asset Prices')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Asset Prices')
    ax2.plot(time,simulated_volatility)
    ax2.set_title('Heston Model Variance Process')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Variance')
    plt.show()

def single_run_heston_volatility_vs_price():
    """
    Simulates asset prices and variances over time using the Heston model for a single run with user-specified parameters.
    """
    num_sims = 1

    # Ask the user if they want to enter their own parameter values
    auto = input("Do you want to enter your own parameter values? (y/n) ").lower()
    if auto == "y":
        # Prompt user for Heston parameter values
        S0 = float(input(f"Enter initial asset price (typical range: varies by asset): "))
        T = float(input(f"Enter time horizon in years (typical range: 0 < T < 10): "))
        r = float(input(f"Enter risk-free interest rate (typical range: 0 < r < 0.1): "))
        num_steps = int(input(f"Enter number of time steps in simulation (typical range: 10 < num_steps < 1000): "))
        kappa = float(input(f"Enter rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10): "))
        theta = float(input(f"Enter long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1): "))
        v0 = float(input(f"Enter initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1): "))
        rho = float(input(f"Enter correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1): "))
        sigma = float(input(f"Enter volatility of volatility (typical range: 0 < sigma < 1): "))
    else:
        # Use example Heston parameter values
        S0 = 100.0             # initial asset price (typical range: varies by asset)
        T = 1.0                # time horizon in years (typical range: 0 < T < 10)
        r = 0.02               # risk-free interest rate (typical range: 0 < r < 0.1)
        num_steps = 252        # number of time steps in simulation (typical range: 10 < num_steps < 1000)
        kappa = 3              # rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10)
        theta = 0.20**2        # long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1)
        v0 = 0.25**2           # initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1)
        rho = 0.7              # correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1)
        sigma = 0.6            # volatility of volatility (typical range: 0 < sigma < 1)
        # Print parameter values
    if auto:
        print("Using example variable values:\n")
        print(f"S0 = {S0}\t\tInitial asset price (typical range: varies by asset)")
        print(f"T = {T}\t\tTime horizon in years (typical range: 0 < T < 10)")
        print(f"r = {r}\t\tRisk-free interest rate (typical range: 0 < r < 0.1)")
        print(f"kappa = {kappa}\t\tRate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10)")
        print(f"theta = {theta:.2f}\t\tLong-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1)")
        print(f"v0 = {v0}\t\tInitial variance under risk-neutral dynamics (typical range: 0 < v0 < 1)")
        print(f"rho = {rho}\t\tCorrelation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1)")
        print(f"sigma = {sigma}\t\tVolatility of volatility (typical range: 0 < sigma < 1)")
    else:
        print("Please enter the following parameters:\n")
        S0 = float(input("Initial asset price (typical range: varies by asset): "))
        T = float(input("Time horizon in years (typical range: 0 < T < 10): "))
        r = float(input("Risk-free interest rate (typical range: 0 < r < 0.1): "))
        kappa = float(input("Rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10): "))
        theta = float(input(f"Long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1) (most typical value: {0.20**2:.2f}): "))
        v0 = float(input("Initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1): "))
        rho = float(input("Correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1): "))
        sigma = float(input("Volatility of volatility (typical range: 0 < sigma < 1): "))

    # simulate asset prices and variances using the Heston model
    simulated_stock_price, simulated_volatility = heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, num_steps, num_sims, r)

    # plot asset prices and variances over time
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.set_title('Heston Model Asset Prices and Variance')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price', color='r')
    ax1.plot(np.linspace(0,T,num_steps+1), simulated_stock_price, color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Variance', color='b')
    ax2.plot(np.linspace(0,T,num_steps+1), simulated_volatility, color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    plt.show()


    # Print parameter values
    print("\nRunning the single_run_heston function to simulate asset prices and variances over time with the following parameter values:\n")
    print(f"S0\t\t{T}\t\t{type(S0)}\t\tInitial asset price (typical range: varies by asset)")
    print(f"T\t\t{T}\t\t{type(T)}\t\tTime horizon in years (typical range: 0 < T < 10)")

def heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, option_type, K): 
    """
    Calculate the price of a European call or put option using the Heston model.
    
    Parameters:
    - S0: initial asset price (depends on asset)
    - v0: initial variance (typical range: 0 < v0 < 1)
    - rho: correlation between asset returns and variance (typical range: -1 < rho < 1)
    - kappa: rate of mean reversion in variance process (typical range: 0 < kappa < 10)
    - theta: long-term mean of variance process (typical range: 0 < theta < 1)
    - sigma: volatility of volatility or the degree of randomness in the variance process (typical range: 0 < sigma < 1)
    - T: time of simulation in years (typical range: 0 < T < 10)
    - num_steps: number of time steps (typical range: 10 < N < 1000)
    - num_sims: number of scenarios/simulations (typical range: 10 < M < 1000)
    - r: risk-free interest rate (typical range: 0 < r < 0.1)
    - option_type: either 'call' or 'put'
    - K: strike price of the option
    
    Returns:
    - The price of the option
    """
    
    # Get the simulated asset prices and variances
    stock_prices, variances = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r)
    
    # Calculate the payoffs for the call or put option
    if option_type == 'call':
        payoffs = np.maximum(stock_prices[-1,:] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - stock_prices[-1,:], 0)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")
    
    # Discount the expected payoffs to the present value
    discounted_payoffs = np.exp(-r*T) * payoffs
    
    # Calculate the option price as the average of the discounted payoffs
    option_price = np.mean(discounted_payoffs)
    
    return option_price


def calculate_put_price():
    """
    Calculates the put price of a Heston option for different parameter inputs and prints the results.
    """
    
    # Define a range of parameter values to test
    v0_vals = [0.05, 0.1, 0.2]
    rho_vals = [-0.5, 0.0, 0.5]
    kappa_vals = [0.5, 1.0, 2.0]
    theta_vals = [0.05, 0.1, 0.2]
    sigma_vals = [0.1, 0.2, 0.3]
    T_vals = [1.0, 2.0, 3.0]
    r_vals = [0.01, 0.05, 0.1]
    K_vals = [90, 100, 110]
    
    # Set other parameters for the Heston model and option type
    S0 = 100
    num_steps = 100
    num_sims = 10000
    option_type = 'put'
    
    # Iterate over parameter combinations and calculate option price
    for v0 in v0_vals:
        for rho in rho_vals:
            for kappa in kappa_vals:
                for theta in theta_vals:
                    for sigma in sigma_vals:
                        for T in T_vals:
                            for r in r_vals:
                                for K in K_vals:
                                    option_price = heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, option_type, K)
                                    print("v0: {}, rho: {}, kappa: {}, theta: {}, sigma: {}, T: {}, r: {}, K: {} - Put option price: {}".format(v0, rho, kappa, theta, sigma, T, r, K, option_price))

calculate_put_price()

def main():
    print("Choose test")
    print("1 = illustrate_heston with example values")
    print("2 = Single run of Heston showing stockprice vs volatility")
    test = int(input(" Choose test: "))
    if test==1:
        illustrate_heston()
    if test==2:
        single_run_heston_volatility_vs_price()
    

    print("The end, thank you ang come again (y)")

#main()