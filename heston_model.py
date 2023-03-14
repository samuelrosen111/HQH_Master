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


################################################## Tool functions

def heston_model_MonteCarlo(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r):
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
    stock_prices, variances = heston_model_MonteCarlo(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r)
    
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

################################################## Illustrative functions

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
    simulated_stock_price, simulated_volatility = heston_model_MonteCarlo(S0, v0, rho, kappa, theta, sigma,T, num_steps, num_sims, r)
    
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

    # Prompt user for Heston parameter values
    auto = input("Do you want to enter your own parameter values? (y/n, default is 'n') ").lower()

    if auto == "y":
        S0 = float(input(f"Enter initial asset price (typical range: varies by asset): "))
        T = float(input(f"Enter time horizon in years (typical range: 0 < T < 10): "))
        r = float(input(f"Enter risk-free interest rate (typical range: 0 < r < 0.1): "))
        num_steps = int(input(f"Enter number of time steps in simulation (typical range: 10 < num_steps < 1000): "))
        kappa = float(input(f"Enter rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10): "))
        theta = float(input(f"Enter long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1) (most typical value: {0.20**2:.2f}): "))
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

    print("\nRunning the single_run_heston function to simulate asset prices and variances over time with the following parameter values:\n")
    print(f"S0\t\t{S0}\t\tInitial asset price (typical range: varies by asset)")
    print(f"T\t\t{T}\t\tTime horizon in years (typical range: 0 < T < 10)")
    print(f"r\t\t{r}\t\tRisk-free interest rate (typical range: 0 < r < 0.1)")
    print(f"kappa\t\t{kappa}\t\tRate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10)")
    print(f"theta\t\t{theta:.2f}\t\tLong-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1)")
    print(f"v0\t\t{v0}\t\tInitial variance under risk-neutral dynamics (typical range: 0 < v0 < 1)")
    print(f"rho\t\t{rho}\t\tCorrelation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1)")
    print(f"sigma\t\t{sigma}\t\tVolatility of volatility (typical range: 0 < sigma < 1)")

    # simulate asset prices and variances using the Heston model
    simulated_stock_price, simulated_volatility = heston_model_MonteCarlo(S0, v0, rho, kappa, theta, sigma,T, num_steps, num_sims, r)

    # plot asset prices and variances over time
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.set_title('Heston Model Asset Prices and Variance')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price', color='r')
    ax1.plot(np.linspace(0,T,num_steps+1), simulated_stock_price, color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volatility', color='b')
    ax2.plot(np.linspace(0,T,num_steps+1), simulated_volatility, color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    plt.show()

def heston_mc_example_run():
    # Set parameter values
    S0 = 100.0     # Initial asset price (typically in the range of $10 to $1000)
    v0 = 0.05      # Initial variance or volatility of the asset price (typically between 0 and 1)
    rho = -0.5     # Correlation between asset returns and variance (typically between -1 and 1)
    kappa = 2.0    # Rate of mean reversion in variance process (typically between 0 and 10)
    theta = 0.05   # Long-term mean of variance process (typically between 0 and 1)
    sigma = 1.0    # Volatility of volatility or degree of randomness in variance process (typically between 0 and 1)
    T = 1.0        # Time to maturity of the option (in years; typically between 0 and 10)
    num_steps = 100  # Number of time steps to simulate (typically between 10 and 1000)
    num_sims = 10000 # Number of Monte Carlo simulations to run (typically between 10 and 1000)
    r = 0.02       # Risk-free interest rate (typically between 0 and 0.1)
    option_type = 'call' # Type of option to price (either 'call' or 'put')
    K = 110.0      # Strike price of the option (typically close to the current asset price)
    
    # Calculate the option price using the Heston model
    option_price = heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, option_type, K)
    
    # Print the input parameters and option price
    print("Heston Option Pricing Example Run")
    print("--------------------------------")
    print("Initial Asset Price (S0): {:.2f}".format(S0))
    print("Initial Variance (v0): {:.4f}".format(v0))
    print("Correlation (rho): {:.2f}".format(rho))
    print("Rate of Mean Reversion (kappa): {:.2f}".format(kappa))
    print("Long-term Mean of Variance (theta): {:.2f}".format(theta))
    print("Volatility of Volatility (sigma): {:.2f}".format(sigma))
    print("Time to Maturity (T): {:.2f}".format(T))
    print("Number of Time Steps (num_steps): {}".format(num_steps))
    print("Number of Simulations (num_sims): {}".format(num_sims))
    print("Risk-free Interest Rate (r): {:.2f}".format(r))
    print("Option Type: {}".format(option_type))
    print("Strike Price (K): {:.2f}".format(K))
    print("Option Price: {:.4f}".format(option_price))

def heston_mc_number_of_timesteps_convergence():
    """    
    This function plots:
    - The error of the Heston_MC method v.s. amount of timesteps
    - Linear best-fit approximation to the error on a log-log scale
    """

    S0 = 100.0         # Initial stock price (depends on asset)
    v0 = 0.05          # Initial variance (typical range: 0 < v0 < 1)
    rho = -0.5         # Correlation between stock returns and variance (typical range: -1 < rho < 1)
    kappa = 2.0        # Mean reversion rate of the variance process (typical range: 0 < kappa < 10)
    theta = 0.05       # Long-term mean of the variance process (typical range: 0 < theta < 1)
    sigma = 1.0        # Volatility of the variance process (volatility of volatility) (typical range: 0 < sigma < 1)
    T = 1.0            # Time to expiration (in years) (typical range: 0 < T < 10)
    num_sims = 10000   # Number of Monte Carlo simulations (typical range: 10 < num_sims < 1000)
    r = 0.02           # Risk-free interest rate (typical range: 0 < r < 0.1)
    option_type = 'call' # Type of the option ('call' or 'put')
    K = 110.0          # Strike price of the option
    good_approx_steps = 1000 # Number of time steps for the "good" approximation (typically large)
    max_steps = 100     # Maximum number of time steps to consider (typically in the range of 10 to 1000)

    # Calculate the "good" approximation using a large number of time steps
    good_approx = heston_option(S0, v0, rho, kappa, theta, sigma, T, good_approx_steps, num_sims, r, option_type, K)
    
    # Initialize arrays to store Heston option prices and errors for different time steps
    heston_values_vs_timestep = np.zeros(max_steps)
    heston_mc_err_vs_timestep = np.zeros(max_steps)
    
    # Loop over different time steps and calculate Heston option prices
    timesteps = np.logspace(1, 3, num=max_steps, dtype=int)
    for i, steps in enumerate(timesteps):
        heston_values_vs_timestep[i] = heston_option(S0, v0, rho, kappa, theta, sigma, T, steps, num_sims, r, option_type, K)
        heston_mc_err_vs_timestep[i] = np.abs(good_approx - heston_values_vs_timestep[i])
    

    # Calculate the linear best-fit approximation to the error array in log-log scale
    log_timesteps = np.log10(timesteps)
    log_errors = np.log10(heston_mc_err_vs_timestep)


    N = len(log_timesteps)
    mean_x = np.mean(log_timesteps)
    mean_y = np.mean(log_errors)
    sum_xy = np.sum((log_timesteps - mean_x) * (log_errors - mean_y))
    sum_x_squared = np.sum((log_timesteps - mean_x)**2)

    slope = sum_xy / sum_x_squared 
    intercept = mean_y - slope * mean_x
    linear_aprox = slope * log_timesteps + intercept
    

    # Print the slope of the linear approximation
    print(f'The slope of the linear approximation is: {slope:.3f}')


    # Plot the error vs. time steps on a log-log chart with linear approximation
    plt.loglog(timesteps, heston_mc_err_vs_timestep, label='Error', color='blue')
    plt.loglog(timesteps, 10**linear_aprox, label='Linear Approximation', color='green')
    plt.title('Error of Heston_MC as function of # timesteps')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def heston_mc_number_of_simulations_convergence():
    """    
    This function plots:
    - The error of the Heston_MC method v.s. number of simulations
    - Linear best-fit approximation to the error on a log-log scale
    """

    S0 = 100.0         # Initial stock price (depends on asset)
    v0 = 0.05          # Initial variance (typical range: 0 < v0 < 1)
    rho = -0.5         # Correlation between stock returns and variance (typical range: -1 < rho < 1)
    kappa = 2.0        # Mean reversion rate of the variance process (typical range: 0 < kappa < 10)
    theta = 0.05       # Long-term mean of the variance process (typical range: 0 < theta < 1)
    sigma = 1.0        # Volatility of the variance process (volatility of volatility) (typical range: 0 < sigma < 1)
    T = 1.0            # Time to expiration (in years) (typical range: 0 < T < 10)
    num_steps = 100    # Number of time steps in simulation (typical range: 10 < num_steps < 1000)
    r = 0.02           # Risk-free interest rate (typical range: 0 < r < 0.1)
    option_type = 'call' # Type of the option ('call' or 'put')
    K = 110.0          # Strike price of the option
    good_approx_sims = 100000 # Number of simulations for the "good" approximation (typically large)
    max_sims = 100     # Maximum number of simulations to consider (typically in the range of 10 to 1000)

    # Calculate the "good" approximation using a large number of simulations
    good_approx = heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, good_approx_sims, r, option_type, K)
    
    # Initialize arrays to store Heston option prices and errors for different numbers of simulations
    heston_values_vs_sims = np.zeros(max_sims)
    heston_mc_err_vs_sims = np.zeros(max_sims)
    
    # Loop over different numbers of simulations and calculate Heston option prices
    num_sims = np.logspace(1, 3, num=max_sims, dtype=int)
    for i, sims in enumerate(num_sims):
        heston_values_vs_sims[i] = heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, sims, r, option_type, K)
        heston_mc_err_vs_sims[i] = np.abs(good_approx - heston_values_vs_sims[i])
    
    # Calculate the linear best-fit approximation to the error array in log-log scale
    log_sims = np.log10(num_sims)
    log_errors = np.log10(heston_mc_err_vs_sims)
    
    # Calculate the linear best-fit approximation to the error array in log-log scale
    log_nsims = np.log10(num_sims)
    log_errors = np.log10(heston_mc_err_vs_sims)

    N = len(log_nsims)
    mean_x = np.mean(log_nsims)
    mean_y = np.mean(log_errors)
    sum_xy = np.sum((log_nsims - mean_x) * (log_errors - mean_y))
    sum_x_squared = np.sum((log_nsims - mean_x)**2)

    slope = sum_xy / sum_x_squared 
    intercept = mean_y - slope * mean_x
    linear_approx = slope * log_nsims + intercept
    
    # Print the slope of the linear approximation
    print(f'The slope of the linear approximation is: {slope:.3f}')
    
    # Plot the error vs. number of simulations on a log-log chart with linear approximation
    plt.loglog(num_sims, heston_mc_err_vs_sims, label='Error', color='blue')
    plt.loglog(num_sims, 10**linear_approx, label='Linear Approximation (slope = {:.3f})'.format(slope), color='green')
    plt.title('Error of Heston MC as function of # simulations')
    plt.xlabel('# Simulations')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

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

def illustrate_heston_Volatility():
    user_choice = input("Would you like to use example parameters or manually enter custom parameters? Enter 'example' or 'custom': ")
    
    if user_choice == 'example':
        # Set parameter values
        v0 = 0.05                 # Initial volatility (typical range: [0.01, 0.1])
        theta = 0.05              # The mean to which the volatility reverts to. (typical range: [0.01, 0.1])
        kappa = 1.5               # Mean reversion speed (typical range: [0.1, 10])
        sigma = 0.3               # Volatility of volatility (typical range: [0.1, 0.5])
        dt = 0.01                 # Time step length (typical range: [0.001, 0.01])
        num_steps = int(1/dt)     # Number of steps, set such that 1/dt is an integer value. Normalize total time-span to 1 for consistency
    else:
        # Prompt the user to manually enter parameter values
        v0 = float(input("Enter the initial volatility (typical range: [0.01, 0.1]): "))
        theta = float(input("Enter the mean to which the volatility reverts to (typical range: [0.01, 0.1]): "))
        kappa = float(input("Enter the mean reversion speed (typical range: [0.1, 10]): "))
        sigma = float(input("Enter the volatility of volatility (typical range: [0.1, 0.5]): "))
        dt = float(input("Enter the time step length (typical range: [0.001, 0.01]): "))
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
        # Plot the volatility over time
    time = np.linspace(0, (num_steps-1)*dt, num_steps)
    plt.plot(time, volatility_over_time, label='Volatility')
    plt.plot(time, [theta for i in range(num_steps)], label='Mean Reversion Level', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title('Heston Model Volatility over Time')
    plt.legend()
    plt.show()

def main():
    print("Choose test")
    print("1: 'illustrate_heston' with example values")
    print("2: Single run of Heston showing stockprice vs volatility")
    print("3: Make one example run of Heston model")
    print("4: Plot heston_mc_number_of_timesteps_convergence")
    print("5: Plot heston_mc_number_of_simulations_convergence")
    print("6: Visualize Heston Volatility")
    user_choice = int(input("\n------> Choose test: "))

    if user_choice==1:
        illustrate_heston()
    if user_choice==2:
        single_run_heston_volatility_vs_price()
    if user_choice==3:
        heston_mc_example_run()
    if user_choice==4:
        heston_mc_number_of_timesteps_convergence()
    if user_choice==5:
        heston_mc_number_of_simulations_convergence()
    if user_choice==6:
        illustrate_heston_Volatility()
    else:
        print("Invalid input")

    print("The end, thank you ang come again (y)")

main()