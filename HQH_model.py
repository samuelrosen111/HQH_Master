import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  # pip install tqdm
import pandas as pd
from time import sleep as time_sleep
import scipy.stats as stats

def calculate_correlation(x, y):
    """
    Calculates the correlation coefficient between two arrays using Pearson's formula.
    
    Args:
    x (array-like): First array.
    y (array-like): Second array.
    
    Returns:
    float: Correlation coefficient between x and y.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    
    corr_coef = numerator / denominator
    
    return corr_coef

################################################## Heston Model

def heston_model_MonteCarlo(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r):
    # Heston_model_MonteCarlo: Simulates asset prices and volatilities using the Heston model via Monte Carlo simulation
    """
    (Credit for correct theory and function implementation: https://quantpy.com.au/stochastic-volatility-models/simulating-heston-model-in-python/)

    Simulate asset prices and variance using the Heston model.
    Parameters:
    - S0: initial asset price (depends on asset)
    - v0: initial variance (typical range: 0 < v0 < 1)
    - rho: correlation between asset returns and variance (typical range: -1 < rho < 1)
    (Rho typically negative high stockprice means low vol usually)
    - kappa: rate of mean reversion in variance process (typical range: 0 < kappa < 10)
    - theta: long-term mean of variance process (typical range: 0 < theta < 1)
    - sigma: volatility of volatility or the degree of randomness in the variance process (typical range: 0 < sigma < 1)
    - T: time of simulation in years (typical range: 0 < T < 10)
    - num_steps: number of time steps (typical range: 10 < N < 1000)
    - num_sims: number of scenarios/simulations (typical range: 10 < M < 1000)
    - r: risk-free interest rate (typical range: 0 < r < 0.1; e.g., 0.1 would be a 10% interest rate)

    
    Returns:
    - numpy array of asset prices over time (shape: (N+1, M))
    - numpy array of variances over time (shape: (N+1, M))
    """

    check_if_parameters_are_okay = hqh_parameters_are_valid()

    # Calculate time increment
    dt = T/num_steps
    
    # Set initial drift and covariance matrix
    drift_term = [0,0]
    covariance_matrix = np.array([[1,rho], [rho,1]])
    
    # Create arrays to store asset prices and variances over time
    asset_price = np.full(shape=(num_steps+1,num_sims), fill_value=S0)
    volatility = np.full(shape=(num_steps+1,num_sims), fill_value=v0)
    
    # Sample correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(drift_term, covariance_matrix, (num_steps,num_sims))
    
    # Calculate asset prices and variances over time, tqdm is used to display
    for i in tqdm(range(1, num_steps + 1), desc="Simulation progress", ncols=100):
        asset_price[i] = asset_price[i - 1] * np.exp((r - 0.5 * volatility[i - 1]) * dt + np.sqrt(volatility[i - 1] * dt) * Z[i - 1, :, 0])
        volatility[i] = np.maximum(volatility[i - 1] + kappa * (theta - volatility[i - 1]) * dt + sigma * np.sqrt(volatility[i - 1] * dt) * Z[i - 1, :, 1], 0)

    return {"asset_prices": asset_price, "volatility": volatility, "params": {"S0": S0, "v0": v0, "rho": rho, "kappa": kappa, "theta": theta, "sigma": sigma, "T": T, "num_steps": num_steps, "num_sims": num_sims, "r": r}}
    # return asset_price, volatility old, non-dynamic return

def heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, option_type, K): 
    # Heston_option: Calculates European option prices using the Heston model and Monte Carlo simulation
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
    - r: risk-free interest rate (typical range: 0 < r < 0.1; e.g., 0.1 would be a 10% interest rate)
    - option_type: type of the option, here is it either 'call' or 'put'
    - K: strike price of the option
    
    Returns:
    - The price of the option
    """    
    # Get the simulated asset prices and variances
    heston_model_results = heston_model_MonteCarlo(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r) #contains asset price, volatility and parameters
    stock_prices = heston_model_results["asset_prices"] #In this funciton only price is of interest

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

def hqh_parameters_are_valid(S0=None, v0=None, rho=None, kappa=None, theta=None, sigma=None, T=None, num_steps=None, num_sims=None, r=None):
    # Error_handling: Validates all Heston-Queue-Hawkes model parameters, detects errors, and prompts user to continue or retry with new input
    errors = []

    if S0 is not None and S0 <= 0:
        errors.append("Initial asset price (S0) must be greater than 0.")
    if v0 is not None and v0 <= 0:
        errors.append("Initial variance (v0) must be greater than 0.")
    if rho is not None and not (-1 <= rho <= 1):
        errors.append("Correlation (rho) must be in the range (-1, 1).")
    if kappa is not None and kappa <= 0:
        errors.append("Rate of mean reversion (kappa) must be greater than 0.")
    if theta is not None and theta <= 0:
        errors.append("Long-term mean of variance (theta) must be greater than 0.")
    if sigma is not None and sigma <= 0:
        errors.append("Volatility of volatility (sigma) must be greater than 0.")
    if T is not None and T <= 0:
        errors.append("Time of simulation (T) must be greater than 0 years.")
    if num_steps is not None and num_steps < 1:
        errors.append("Number of time steps (num_steps) cannot be less than 1")
    if num_sims is not None and num_sims < 1:
        errors.append("Number of simulations (num_sims) cannot be less than 1")
    if r is not None and r < 0:
        errors.append("Risk-free interest rate (r) must be non-negative.")


    if errors:
        print("The following errors were detected:")
        for error in errors:
            print(f"- {error}")

        user_input = input("Do you want to continue despite these errors? (y/n): ").lower()
        if user_input == 'y':
            print("Continuing with the given parameters...")
            return False
        else:
            print("Exiting the program.")
            exit(0)
    else:
        return True

################################################## Hawkes Process

def exponential_excitation(t, alpha, beta):
    """
    Exponential excitation function.
    
    Parameters:
    - t: time since the last event
    - alpha: positive constant affecting the intensity of the excitation
    - beta: positive constant affecting the decay rate of the excitation
    
    Returns:
    - Excitation value at time t
    """
    return alpha * np.exp(-beta * t)

def hawkes_process(mu, alpha, beta, T):
    """
    Simulate a Hawkes process with an exponential excitation function.
    Parameters:
    - mu: baseline intensity
    - alpha: positive constant affecting the intensity of the excitation
    - beta: positive constant affecting the decay rate of the excitation
    - T: time period to simulate the process
    Returns:
    - events: list of event times
    """
    events = []
    t = 0
    
    while t <= T:
        lambda_max = mu + alpha  # Upper bound on intensity (assuming exponential excitation)
        t += -np.log(np.random.rand()) / lambda_max  # Generate inter-event time from an exponential distribution
        
        # Acceptance-rejection sampling
        p = mu / lambda_max
        if len(events) > 0:
            p += exponential_excitation(t - events[-1], alpha, beta) / lambda_max
        
        if np.random.rand() < p:
            events.append(t)
    
    return events

################################################## Heston Model + Hawkes self excitation

def heston_hawkes_parameters_are_valid(S0=None, v0=None, rho=None, kappa=None, theta=None, sigma=None, T=None, num_steps=None, num_sims=None, r=None, hawkes_mu=None, hawkes_alpha=None, hawkes_beta=None):
    # Error_handling: Validates Heston model parameters, detects errors, and prompts user to continue or retry with new input
    errors = []

    if S0 is not None and S0 <= 0:
        errors.append("Initial asset price (S0) must be greater than 0.")
    if v0 is not None and v0 <= 0:
        errors.append("Initial variance (v0) must be greater than 0.")
    if rho is not None and not (-1 <= rho <= 1):
        errors.append("Correlation (rho) must be in the range (-1, 1).")
    if kappa is not None and kappa <= 0:
        errors.append("Rate of mean reversion (kappa) must be greater than 0.")
    if theta is not None and theta <= 0:
        errors.append("Long-term mean of variance (theta) must be greater than 0.")
    if sigma is not None and sigma <= 0:
        errors.append("Volatility of volatility (sigma) must be greater than 0.")
    if T is not None and T <= 0:
        errors.append("Time of simulation (T) must be greater than 0 years.")
    if num_steps is not None and num_steps < 1:
        errors.append("Number of time steps (num_steps) cannot be less than 1")
    if num_sims is not None and num_sims < 1:
        errors.append("Number of simulations (num_sims) cannot be less than 1")
    if r is not None and r < 0:
        errors.append("Risk-free interest rate (r) must be non-negative.")
    if hawkes_mu < 0:
        errors.append("Background intensity mu must be positive.")
    if hawkes_alpha < 0:
        errors.append("Past event influence alpha must be positive.")
    if hawkes_beta <=0:
        errors.append("Decay speed beta must have a positive value, otherwise one gets divergence.")

    if errors:
        print("The following errors were detected:")
        for error in errors:
            print(f"- {error}")

        user_input = input("Do you want to continue despite these errors? (y/n): ").lower()
        if user_input == 'y':
            print("Continuing with the given parameters...")
            return False
        else:
            print("Exiting the program.")
            exit(0)
    else:
        return True

def heston_hawkes(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta):
    # Heston_model_MonteCarlo: Simulates asset prices and volatilities using the Heston model via Monte Carlo simulation
    """
    (Credit for correct theory and function implementation: https://quantpy.com.au/stochastic-volatility-models/simulating-heston-model-in-python/)
    Simulate asset prices and variance using the Heston model.
    Parameters:
    - S0: initial asset price (depends on asset)
    - v0: initial variance (typical range: 0 < v0 < 1)
    - rho: correlation between asset returns and variance (typical range: -1 < rho < 1)
    (Rho typically negative high stockprice means low vol usually)
    - kappa: rate of mean reversion in variance process (typical range: 0 < kappa < 10)
    - theta: long-term mean of variance process (typical range: 0 < theta < 1)
    - sigma: volatility of volatility or the degree of randomness in the variance process (typical range: 0 < sigma < 1)
    - T: time of simulation in years (typical range: 0 < T < 10)
    - num_steps: number of time steps (typical range: 10 < N < 1000)
    - num_sims: number of scenarios/simulations (typical range: 10 < M < 1000)
    - r: risk-free interest rate (typical range: 0 < r < 0.1; e.g., 0.1 would be a 10% interest rate)
    Returns:
    - numpy array of asset prices over time (shape: (N+1, M))
    - numpy array of variances over time (shape: (N+1, M))
    """

    event_times = hawkes_process(hawkes_mu, hawkes_alpha, hawkes_beta, T)

    # Calculate time increment
    dt = T/num_steps
    
    # Set initial drift and covariance matrix
    drift_term = [0,0]
    covariance_matrix = np.array([[1,rho], [rho,1]])
    
    # Create arrays to store asset prices and variances over time
    asset_price = np.full(shape=(num_steps+1,num_sims), fill_value=S0)
    asset_volatility = np.full(shape=(num_steps+1,num_sims), fill_value=v0)
    
    # Sample correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(drift_term, covariance_matrix, (num_steps,num_sims))
    
    event_impact_on_volatility = 1.05
    event_impact_on_price = 1.05

    current_event_index = 0
    next_event_time = event_times[current_event_index] if event_times else None

    for i in tqdm(range(1, num_steps + 1), desc="Simulation progress", ncols=100):
        current_time = i * dt

        asset_price[i] = asset_price[i - 1] * np.exp((r - 0.5 * asset_volatility[i - 1]) * dt + np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 0])
        asset_volatility[i] = np.maximum(asset_volatility[i - 1] + kappa * (theta - asset_volatility[i - 1]) * dt + sigma * np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 1], 0)

        # Check if an event occurred at the current time step
        while next_event_time is not None and current_time >= next_event_time:
            asset_price[i] *= event_impact_on_price
            asset_volatility[i] *= event_impact_on_volatility
            
            current_event_index += 1
            if current_event_index < len(event_times):
                next_event_time = event_times[current_event_index]
            else:
                next_event_time = None
        

    return asset_price, asset_volatility, event_times


################################################## Heston Hawkes Model with normal distrbuted impacts

def generate_impacts_from_events(event_times, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact):
    """
    Generate two vectors of impacts based on a given events vector, with impact values drawn from two different normal distributions.
    
    Parameters:
    - event_arrival_times: list of event times
    - mean_price_impact: mean of the normal distribution for price impacts
    - std_price_impact: standard deviation of the normal distribution for price impacts
    - mean_vol_impact: mean of the normal distribution for volatility impacts
    - std_vol_impact: standard deviation of the normal distribution for volatility impacts

    Returns:
    - price_impacts: list of price impacts corresponding to the events
    - volatility_impacts: list of volatility impacts corresponding to the events
    """
    num_events = len(event_times)
    price_impacts = np.random.normal(mean_price_impact, std_price_impact, num_events)
    volatility_impacts = np.random.normal(mean_vol_impact, std_vol_impact, num_events)
    return price_impacts, volatility_impacts

def heston_hawkes_normal_mc_test(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact):
    # Heston_model_MonteCarlo: Simulates asset prices and volatilities using the Heston model via Monte Carlo simulation
    """
    (Credit for correct theory and function implementation: https://quantpy.com.au/stochastic-volatility-models/simulating-heston-model-in-python/)
    Simulate asset prices and variance using the Heston model.
    Parameters:
    - S0: initial asset price (depends on asset)
    - v0: initial variance (typical range: 0 < v0 < 1)
    - rho: correlation between asset returns and variance (typical range: -1 < rho < 1)
    (Rho typically negative high stockprice means low vol usually)
    - kappa: rate of mean reversion in variance process (typical range: 0 < kappa < 10)
    - theta: long-term mean of variance process (typical range: 0 < theta < 1)
    - sigma: volatility of volatility or the degree of randomness in the variance process (typical range: 0 < sigma < 1)
    - T: time of simulation in years (typical range: 0 < T < 10)
    - num_steps: number of time steps (typical range: 10 < N < 1000)
    - num_sims: number of scenarios/simulations (typical range: 10 < M < 1000)
    - r: risk-free interest rate (typical range: 0 < r < 0.1; e.g., 0.1 would be a 10% interest rate)
    Returns:
    - numpy array of asset prices over time (shape: (N+1, M))
    - numpy array of variances over time (shape: (N+1, M))
    """

    event_times = hawkes_process(hawkes_mu, hawkes_alpha, hawkes_beta, T)
    price_impacts, volatility_impacts = generate_impacts_from_events(event_times, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact)


    # Calculate time increment
    dt = T/num_steps
    
    # Set initial drift and covariance matrix
    drift_term = [0,0]
    covariance_matrix = np.array([[1,rho], [rho,1]])
    
    # Create arrays to store asset prices and variances over time
    asset_price = np.full(shape=(num_steps+1,num_sims), fill_value=S0)
    asset_volatility = np.full(shape=(num_steps+1,num_sims), fill_value=v0)
    
    # Sample correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(drift_term, covariance_matrix, (num_steps,num_sims))
    
    event_impact_on_volatility = 1.05
    event_impact_on_price = 1.05

    current_event_index = 0
    next_event_time = event_times[current_event_index] if event_times else None

    for i in tqdm(range(1, num_steps + 1), desc="Simulation progress", ncols=100):
        current_time = i * dt

        asset_price[i] = asset_price[i - 1] * np.exp((r - 0.5 * asset_volatility[i - 1]) * dt + np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 0])
        asset_volatility[i] = np.maximum(asset_volatility[i - 1] + kappa * (theta - asset_volatility[i - 1]) * dt + sigma * np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 1], 0)

        # Check if an event occurred at the current time step
        while next_event_time is not None and current_time >= next_event_time:
            asset_price[i] *= event_impact_on_price
            asset_volatility[i] *= event_impact_on_volatility
            
            current_event_index += 1
            if current_event_index < len(event_times):
                next_event_time = event_times[current_event_index]
            else:
                next_event_time = None
        

    return asset_price, asset_volatility, event_times

def heston_hawkes_normal_mc(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact):
    
    """
    Simulate asset prices and variance using the Heston model, incorporating event impacts from a Hawkes process with impact values drawn from two different normal distributions.

    Parameters:
    - S0: initial asset price
    - v0: initial variance
    - rho: correlation between asset returns and variance
    - kappa: rate of mean reversion in variance process
    - theta: long-term mean of variance process
    - sigma: volatility of volatility, degree of randomness in the variance process
    - T: time of simulation in years
    - num_steps: number of time steps
    - num_sims: number of scenarios/simulations
    - r: risk-free interest rate
    - hawkes_mu: baseline intensity for Hawkes process
    - hawkes_alpha: positive constant affecting the intensity of the excitation for Hawkes process
    - hawkes_beta: positive constant affecting the decay rate of the excitation for Hawkes process
    - mean_price_impact: mean of the normal distribution for price impacts
    - std_price_impact: standard deviation of the normal distribution for price impacts
    - mean_vol_impact: mean of the normal distribution for volatility impacts
    - std_vol_impact: standard deviation of the normal distribution for volatility impacts

    Returns:
    - asset_price: numpy array of asset prices over time (shape: (num_steps+1, num_sims))
    - asset_volatility: numpy array of variances over time (shape: (num_steps+1, num_sims))
    - event_times: list of event times from the Hawkes process
    """

    event_times = hawkes_process(hawkes_mu, hawkes_alpha, hawkes_beta, T)
    price_impacts, volatility_impacts = generate_impacts_from_events(event_times, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact)

    dt = T/num_steps
    drift_term = [0,0]
    covariance_matrix = np.array([[1,rho], [rho,1]])
    
    asset_price = np.full(shape=(num_steps+1,num_sims), fill_value=S0)
    asset_volatility = np.full(shape=(num_steps+1,num_sims), fill_value=v0)
    
    Z = np.random.multivariate_normal(drift_term, covariance_matrix, (num_steps,num_sims))

    current_event_index = 0
    next_event_time = event_times[current_event_index] if event_times else None

    for i in tqdm(range(1, num_steps + 1), desc="Simulation progress", ncols=100):
        current_time = i * dt

        asset_price[i] = asset_price[i - 1] * np.exp((r - 0.5 * asset_volatility[i - 1]) * dt + np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 0])
        asset_volatility[i] = np.maximum(asset_volatility[i - 1] + kappa * (theta - asset_volatility[i - 1]) * dt + sigma * np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 1], 0)

        while next_event_time is not None and current_time >= next_event_time:
            asset_price[i] *= price_impacts[current_event_index]
            asset_volatility[i] *= volatility_impacts[current_event_index]
            
            current_event_index += 1
            if current_event_index < len(event_times):
                next_event_time = event_times[current_event_index]
            else:
                next_event_time = None

    return asset_price, asset_volatility, event_times, price_impacts, volatility_impacts


################################################## QUEUE HAWKES wipe


def q_hawkes_process(mu, alpha, T, dt, lambda_param, max_intensity):
    """
    Input parameters:
    - mu: Baseline intensity
    - alpha: Excitation constant affecting the intensity of the excitation
    - T: Total runtime given in years. Should be an integer value.
    - dt: Time step length. Should evenly divide T.
    - lambda_param: Lambda parameter for the exponential distribution.
    - max_intensity: Maximum intensity allowed. Stochastic removal of the queue depends on this parameter.

    Output:
    - intensities: List of intensities at each time step.
    - event_times: List of event occurrence times.

    Function steps:
    1. Initialize the current_intensity, intensities, memory_kernel, and event_times lists.
    2. Iterate through each time step and update the current intensity.
    3. Check if an event occurs in the current time step and update the memory_kernel accordingly.
    4. Update the duration of events and remove expired ones from the memory_kernel.
    5. Check for stochastic memory loss and reset the memory_kernel if needed.
    """

    # Step 1: Initialize the current_intensity, intensities, memory_kernel, and event_times lists.
    current_intensity = mu
    intensities = []
    memory_kernel = []
    event_times = []
    wipe_times = []

    # Step 2: Iterate through each time step and update the current intensity.
    for current_time in np.arange(0, float(T), dt):
        
        current_intensity = mu + alpha * len(memory_kernel)
        intensities.append(current_intensity)

        # Step 3: Check if an event occurs in the current time step and update the memory_kernel accordingly.
        event_occured_this_time_step = np.random.rand() <= current_intensity
        if event_occured_this_time_step:
            event_times.append(current_time)
            duration_time_for_this_event = np.random.exponential(scale=1 / lambda_param)
            memory_kernel.append(duration_time_for_this_event)

        # Step 4: Update the duration of events and remove expired ones from the memory_kernel.
        for i in reversed(range(len(memory_kernel))):
            memory_kernel[i] -= dt
            if memory_kernel[i] <= 0:
                del memory_kernel[i]

        # Step 5: Check for stochastic memory loss and reset the memory_kernel if needed.
        base_intensity = mu
        wipe_probability = (current_intensity - base_intensity) / (max_intensity - base_intensity)
        wipe_event_occured_this_time_step = (np.random.rand() < wipe_probability)
        if wipe_event_occured_this_time_step:
            wipe_times.append(current_time)
            memory_kernel = []

    return intensities, event_times, wipe_times

def run_and_plot_q_hawkes():
    mu = 0.1
    alpha = 0.005
    T = 10
    dt = 0.01
    lambda_param = 0.5
    max_intensity = 1

    intensities, event_times, wipe_times = q_hawkes_process(mu, alpha, T, dt, lambda_param, max_intensity)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(0, float(T), dt), intensities, label='Intensity')
    plt.scatter(event_times, [0]*len(event_times), color='orange', marker='o', label='Events')
    
    for wt in wipe_times:
        plt.axvline(wt, color='purple', linestyle='--', alpha=0.7, label='Wipe Events' if wt == wipe_times[0] else None)
    
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Q-Hawkes Process Simulation')
    plt.legend(loc='upper right')
    plt.show()


def heston_queue_hawkes_normal_mc(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact):
    
    """
    Simulate asset prices and variance using the Heston model, incorporating event impacts from a Hawkes process with impact values drawn from two different normal distributions.

    q_hawkes_process: Generates a Hawkes process with queueing events and memory wipe events.
    generate_impacts_from_events: Generates price and volatility impacts from the event times.

    Parameters:
    - S0: initial asset price
    - v0: initial variance
    - rho: correlation between asset returns and variance
    - kappa: rate of mean reversion in variance process
    - theta: long-term mean of variance process
    - sigma: volatility of volatility, degree of randomness in the variance process
    - T: time of simulation in years
    - num_steps: number of time steps
    - num_sims: number of scenarios/simulations
    - r: risk-free interest rate
    - hawkes_mu: baseline intensity for Hawkes process
    - hawkes_alpha: positive constant affecting the intensity of the excitation for Hawkes process
    - hawkes_beta: positive constant affecting the decay rate of the excitation for Hawkes process
    - mean_price_impact: mean of the normal distribution for price impacts
    - std_price_impact: standard deviation of the normal distribution for price impacts
    - mean_vol_impact: mean of the normal distribution for volatility impacts
    - std_vol_impact: standard deviation of the normal distribution for volatility impacts

    Returns:
    - asset_price: numpy array of asset prices over time (shape: (num_steps+1, num_sims))
    - asset_volatility: numpy array of variances over time (shape: (num_steps+1, num_sims))
    - event_times: list of event times from the Hawkes process
    - price_impacts: list of price impacts corresponding to event times
    - volatility_impacts: list of volatility impacts corresponding to event times
    - wipe_times: list of memory wipe event times

    Steps:
    1. Generate a Hawkes process with queueing events and memory wipe events.
    2. Generate price and volatility impacts from the event times.
    3. Initialize asset price and volatility arrays.
    4. Simulate asset prices and volatility for each time step and scenario, incorporating event impacts when necessary.
    """
    # Step 1: Generate a Hawkes process with queueing events and memory wipe events.
    qh_mu = 0.1
    qh_alpha = 0.005
    qh_dt = 0.01
    qh_lambda_param = 0.5
    qh_max_intensity = 1
    intensities, event_times, wipe_times = q_hawkes_process(qh_mu, qh_alpha, T, qh_dt, qh_lambda_param, qh_max_intensity)
    
    # Step 2: Generate price and volatility impacts from the event times.
    price_impacts, volatility_impacts = generate_impacts_from_events(event_times, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact)

    # Step 3: Initialize asset price and volatility arrays.
    dt = T/num_steps
    drift_term = [0,0]
    covariance_matrix = np.array([[1,rho], [rho,1]])
    
    asset_price = np.full(shape=(num_steps+1,num_sims), fill_value=S0)
    asset_volatility = np.full(shape=(num_steps+1,num_sims), fill_value=v0)
    
    Z = np.random.multivariate_normal(drift_term, covariance_matrix, (num_steps,num_sims))

    current_event_index = 0
    next_event_time = event_times[current_event_index] if event_times else None
    
    #4. Simulate asset prices and volatility for each time step and scenario, incorporating event impacts when necessary.
    for i in tqdm(range(1, num_steps + 1), desc="Simulation progress", ncols=100):
        current_time = i * dt

        asset_price[i] = asset_price[i - 1] * np.exp((r - 0.5 * asset_volatility[i - 1]) * dt + np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 0])
        asset_volatility[i] = np.maximum(asset_volatility[i - 1] + kappa * (theta - asset_volatility[i - 1]) * dt + sigma * np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 1], 0)

        while next_event_time is not None and current_time >= next_event_time:
            asset_price[i] *= price_impacts[current_event_index]
            asset_volatility[i] *= volatility_impacts[current_event_index]
            
            current_event_index += 1
            if current_event_index < len(event_times):
                next_event_time = event_times[current_event_index]
            else:
                next_event_time = None

    return asset_price, asset_volatility, event_times, price_impacts, volatility_impacts, wipe_times

def illustrate_heston_queue_hawkes_normal_mc_without_wipelines():
    # Use example parameter values
    S0 = 100.0
    v0 = 0.25**2
    rho = 0.7
    kappa = 3
    theta = 0.20**2
    sigma = 0.6
    T = 20
    num_steps = 1000
    num_sims = 1
    r = 0.02
    hawkes_mu = 1
    hawkes_alpha = 0.8
    hawkes_beta = 1.5
    mean_price_impact = 1.0
    std_price_impact = 0.1
    mean_vol_impact = 1.0
    std_vol_impact = 0.1

    asset_price, asset_volatility, event_times, price_impacts, volatility_impacts, wipe_times = heston_queue_hawkes_normal_mc(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_title('Heston-Queue-Hawkes Model: Asset Prices v.s. Volatility')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price', color='r')
    ax1.plot(np.linspace(0, T, num_steps + 1), asset_price, color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax1.annotate(
        '(Horizontal purple lines indicate an event occurring)',
        xy=(0.5, -0.1),
        xycoords='axes fraction',
        fontsize=10,
        color='purple',
        ha='center',
        va='top',
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel('Volatility', color='b')
    ax2.plot(np.linspace(0, T, num_steps + 1), asset_volatility, color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Plot the event times as horizontal purple dotted lines
    for event_time in event_times:
        ax1.axvline(event_time, color='purple', linestyle=(0, (3, 3)))

    plt.show()

    # Print the mean and standard deviation of the normal distributions
    print("\n\nAn impact of 1 would mean mutiplying the number by 1 and thus no impact. 1.1 woul mean increasing the value by 10 percent. \nFollowing are the impact values drawn from the normal distribution of both volatility and price: \n")
    print(f"Mean Price Impact: {mean_price_impact:.2f}, Std Dev Price Impact: {std_price_impact:.2f}")
    print(f"Mean Volatility Impact: {mean_vol_impact:.2f}, Std Dev Volatility Impact: {std_vol_impact:.2f}\n")

    # Print the table header
    print("Event Time\tPrice Impact (%)\tVolatility Impact (%)")

    # Print the table rows
    for event_time, price_impact, vol_impact in zip(event_times, price_impacts, volatility_impacts):
        print(f"{event_time:.2f}\t\t{(price_impact - 1) * 100:.2f}\t\t\t{(vol_impact - 1) * 100:.2f}")

    print("\n\n\n")

def illustrate_heston_queue_hawkes_normal_mc():
    # Use example parameter values
    S0 = 100.0
    v0 = 0.25**2
    rho = 0.7
    kappa = 3
    theta = 0.20**2
    sigma = 0.6
    T = 20
    num_steps = 1000
    num_sims = 1
    r = 0.02
    hawkes_mu = 1
    hawkes_alpha = 0.8
    hawkes_beta = 1.5
    mean_price_impact = 1.0
    std_price_impact = 0.1
    mean_vol_impact = 1.0
    std_vol_impact = 0.1

    asset_price, asset_volatility, event_times, price_impacts, volatility_impacts, wipe_times = heston_queue_hawkes_normal_mc(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_title('Heston-Queue-Hawkes Model: Asset Prices v.s. Volatility')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price', color='r')
    ax1.plot(np.linspace(0, T, num_steps + 1), asset_price, color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax1.annotate(
        'Horizontal purple lines indicate an event occurring & Green lines indicate wipe events.',
        xy=(0.5, -0.1),
        xycoords='axes fraction',
        fontsize=10,
        color='purple',
        ha='center',
        va='top',
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel('Volatility', color='b')
    ax2.plot(np.linspace(0, T, num_steps + 1), asset_volatility, color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Plot the event times as horizontal purple dotted lines
    for event_time in event_times:
        ax1.axvline(event_time, color='purple', linestyle=(0, (3, 3)))

    # Plot wipe events as solid green vertical lines
    for wipe_time in wipe_times:
        ax1.axvline(wipe_time, color='green', linestyle='solid')

    plt.show()

    # Print the mean and standard deviation of the normal distributions
    print("\n\nAn impact of 1 would mean mutiplying the number by 1 and thus no impact. 1.1 woul mean increasing the value by 10 percent. \nFollowing are the impact values drawn from the normal distribution of both volatility and price: \n")
    print(f"Mean Price Impact: {mean_price_impact:.2f}, Std Dev Price Impact: {std_price_impact:.2f}")
    print(f"Mean Volatility Impact: {mean_vol_impact:.2f}, Std Dev Volatility Impact: {std_vol_impact:.2f}\n")

     # Print the table header
    print("Event Time\tPrice Impact (%)\tVolatility Impact (%)")

    # Print the table rows
    for event_time, price_impact, vol_impact in zip(event_times, price_impacts, volatility_impacts):
        print(f"{event_time:.2f}\t\t{(price_impact - 1) * 100:.2f}\t\t\t{(vol_impact - 1) * 100:.2f}")

    print("\n\n\n")

#illustrate_heston_queue_hawkes_normal_mc()


################################################## QUEUE HAWKES Samuel 2.0


def q_hawkes_process_dampening(mu, alpha, beta, T, dt, lambda_param, max_intensity):
    """
    Input parameters:
    - mu: Baseline intensity
    - alpha: Excitation constant affecting the intensity of the excitation
    - beta: expirations rate determining speed with which memory of past events is stochastically forgotten
    - T: Total runtime given in years. Should be an integer value.
    - dt: Time step length. Should evenly divide T.
    - lambda_param: Lambda parameter for the exponential distribution.
    - max_intensity: Maximum intensity allowed. Stochastic removal of the queue depends on this parameter.

    Output:
    - intensities: List of intensities at each time step.
    - event_times: List of event occurrence times.

    Function steps:
    1. Initialize the current_intensity, intensities, memory_kernel, and event_times lists.
    2. Iterate through each time step and update the current intensity.
    3. Check if an event occurs in the current time step and update the memory_kernel accordingly.
    4. Update the duration of events and remove expired ones from the memory_kernel.
    5. Check for stochastic memory loss and reset the memory_kernel if needed.
    """

    # Step 1: Initialize the current_intensity, intensities, memory_kernel, and event_times lists.
    current_intensity = mu
    intensities = []
    memory_kernel = []
    event_times = []
    stochastic_memory_loss_times = []

    # Step 2: Iterate through each time step and update the current intensity.
    for current_time in np.arange(0, float(T), dt):
        
        current_intensity = mu + alpha * len(memory_kernel)
        intensities.append(current_intensity)

        # Step 3: Check if an event occurs in the current time step and update the memory_kernel accordingly.
        event_occured_this_time_step = np.random.rand() <= current_intensity
        if event_occured_this_time_step:
            event_times.append(current_time)
            duration_time_for_this_event = np.random.exponential(scale=1 / lambda_param)
            memory_kernel.append(duration_time_for_this_event)

        # Step 4: Update the duration of events and remove expired ones from the memory_kernel.
        for i in reversed(range(len(memory_kernel))):
            memory_kernel[i] -= dt
            if memory_kernel[i] <= 0:
                del memory_kernel[i]

        # Step 5: Check for stochastic memory loss - potentially decreasing the memory_kernel
        base_intensity = mu
        probability_stochastic_memory_loss = (current_intensity - base_intensity) / (max_intensity - base_intensity)
        wipe_event_occured_this_time_step = (np.random.rand() < probability_stochastic_memory_loss)
        if wipe_event_occured_this_time_step:
            stochastic_memory_loss_times.append(current_time)
            for i in range(0,len(memory_kernel)):
                memory_kernel[i] *= beta

    return intensities, event_times, stochastic_memory_loss_times

def run_and_plot_q_hawkes2():
    mu = 0.1
    alpha = 0.005
    beta = 0.9
    T = 10
    dt = 0.01
    lambda_param = 0.5
    max_intensity = 0.3

    intensities, event_times, wipe_times = q_hawkes_process_dampening(mu, alpha, beta, T, dt, lambda_param, max_intensity)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(0, float(T), dt), intensities, label='Intensity')

    # Scatter plot with event intensities
    wipe_intensities = [intensities[np.where(np.arange(0, float(T), dt) == wt)[0][0]] for wt in wipe_times]
    plt.scatter(wipe_times, wipe_intensities, color='purple', marker='x', label='Wipe Events', zorder=3, alpha=0.7)

    # Event markers on the intensity curve
    plt.scatter(event_times, [0]*len(event_times), color='orange', marker='o', label='Events')

    # Plot baseline intensity (mu) as a horizontal line
    plt.axhline(mu, color='green', linestyle='--', label='Baseline Intensity (mu)')

    # Calculate and plot average intensity for the full run
    avg_intensity = sum(intensities) / len(intensities)
    plt.axhline(avg_intensity, color='red', linestyle='--', label='Average Intensity')

    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Q-Hawkes Process Simulation')
    plt.legend(loc='upper right')
    plt.show()

################################################## HESTON QUEUE HAWKES Osterlee

def q_hawkes_process_osterlee(mu, alpha, beta, T, dt):
    """
    Simulates the Q-Hawkes process.

    Parameters:
    - mu (float): Baseline intensity (lambda^* in the paper).
    - alpha (float): Excitation constant affecting the intensity of the excitation.
    - beta (float): Expiration rate determining speed with which memory of past events is stochastically forgotten.
    - T (int): Total runtime given in years.
    - dt (float): Time step length. Should evenly divide T.

    Returns:
    - event_times (list): List of event occurrence times.
    - memory_loss_times (list): List of memory loss occurrence times.
    - event_intensities (list): List of intensities at each time step for events.
    - memory_loss_intensities (list): List of intensities at each time step for stochastic memory loss.
    - Q_values (list): List of activation numbers at each time step.
    """

    # Initialize variables
    Q = 0  # Activation number, higher values increase likelihood of self-excitation and stochastic memory loss
    event_intensities = []  # Likelihood of events happening throughout the simulation (for visualization purposes only)
    memory_loss_intensities = []  # Likelihood of stochastic memory loss happening throughout the simulation (for visualization purposes only)
    event_times = []  # Memory kernel of occurred events
    memory_loss_times = []  # Memory kernel of memory loss occurrences
    Q_values = []

    # Iterate through each time step and update the two affine processes
    for current_time in np.arange(0, float(T), dt):
        # Check if event occurred
        current_event_intensity = mu + alpha * Q  # Current intensity (likelihood) of an event occurring
        event_intensities.append(current_event_intensity)
        if current_event_intensity >= np.random.rand():  # If event occurs
            Q += 1  # Increase the activation number by 1
            event_times.append(current_time)

        # Check if stochastic memory loss should be applied
        current_stochastic_memory_loss_intensity = beta * Q  # Current stochastic probability of memory loss
        memory_loss_intensities.append(current_stochastic_memory_loss_intensity)
        if current_stochastic_memory_loss_intensity >= np.random.rand():  # Check if N_t^Q jumps, if so, apply stochastic memory loss
            Q -= 1  # Decrease the activation number by 1
            memory_loss_times.append(current_time)
        Q_values.append(Q)

        if Q < 0: # Q can't go below zero
            Q = 0

    return event_times, memory_loss_times, event_intensities, memory_loss_intensities, Q_values

def plot_q_hawkes_process_osterlee():
    """
    Simulates and plots the Q-Hawkes process with pre-defined parameters.

    No input required. 
    """
    # Example parameters (change these to suit your needs)
    mu = 0.2  # Baseline intensity (lambda^* in the paper). Typical values might range from 0 to 1.
    alpha = 0.5  # Excitation constant affecting the intensity of the excitation. Typical values might range from 0 to 1.
    beta = 0.6  # Expiration rate determining speed with which memory of past events is stochastically forgotten. Typical values might range from 0 to 1.
    T = 1  # Total runtime given in years. Typical values might range from 1 to 10.
    dt = 0.01  # Time step length. Should evenly divide T. Typical values might be 0.01 or 0.001.

    event_times, memory_loss_times, event_intensities, memory_loss_intensities, Q_values = q_hawkes_process_osterlee(mu, alpha, beta, T, dt)

    # Create time array for x-axis
    time_array = np.arange(0, float(T), dt)

    # Create figure and axes
    fig, axs = plt.subplots(2, sharex=True)

    # Plot event and memory loss intensities
    axs[0].plot(time_array, event_intensities, color='blue', label='Event intensity')
    axs[0].plot(time_array, memory_loss_intensities, color='red', label='Memory loss intensity')

    # Plot average event and memory loss intensities
    axs[0].axhline(np.mean(event_intensities), color='blue', linestyle='dotted', label='Average event intensity')
    axs[0].axhline(np.mean(memory_loss_intensities), color='red', linestyle='dotted', label='Average memory loss intensity')

    # Plot event times and memory loss times
    for event_time in event_times:
        axs[0].plot(event_time, mu + alpha * event_times.count(event_time), 'bx', alpha=0.5)
    for memory_loss_time in memory_loss_times:
        axs[0].plot(memory_loss_time, beta * memory_loss_times.count(memory_loss_time), 'ro', alpha=0.5)

    axs[0].legend()

    # Plot Q values
    axs[1].plot(time_array, Q_values, color='purple', label='Q values')

    # Plot average Q values
    axs[1].axhline(np.mean(Q_values), color='purple', linestyle='dotted', label='Average Q value')

    axs[1].legend()

    plt.show()
 
def heston_queue_hawkes_normal_osterlee(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact, event_times):
    
    """
    Simulate asset prices and variance using the Heston model, incorporating event impacts from a Hawkes process with impact values drawn from two different normal distributions.

    q_hawkes_process: Generates a Hawkes process with queueing events and memory wipe events.
    generate_impacts_from_events: Generates price and volatility impacts from the event times.

    Parameters:
    - S0: initial asset price
    - v0: initial variance
    - rho: correlation between asset returns and variance
    - kappa: rate of mean reversion in variance process
    - theta: long-term mean of variance process
    - sigma: volatility of volatility, degree of randomness in the variance process
    - T: time of simulation in years
    - num_steps: number of time steps
    - num_sims: number of scenarios/simulations
    - r: risk-free interest rate
    - hawkes_mu: baseline intensity for Hawkes process
    - hawkes_alpha: positive constant affecting the intensity of the excitation for Hawkes process
    - hawkes_beta: positive constant affecting the decay rate of the excitation for Hawkes process
    - mean_price_impact: mean of the normal distribution for price impacts
    - std_price_impact: standard deviation of the normal distribution for price impacts
    - mean_vol_impact: mean of the normal distribution for volatility impacts
    - std_vol_impact: standard deviation of the normal distribution for volatility impacts

    Returns:
    - asset_price: numpy array of asset prices over time (shape: (num_steps+1, num_sims))
    - asset_volatility: numpy array of variances over time (shape: (num_steps+1, num_sims))
    - event_times: list of event times from the Hawkes process
    - price_impacts: list of price impacts corresponding to event times
    - volatility_impacts: list of volatility impacts corresponding to event times
    - wipe_times: list of memory wipe event times

    Steps:
    1. Generate price and volatility impacts from the event times.
    2. Initialize asset price and volatility arrays.
    3. Simulate asset prices and volatility for each time step and scenario, incorporating event impacts when necessary.
    """

    
    # Step 2: Generate price and volatility impacts from the event times.
    price_impacts, volatility_impacts = generate_impacts_from_events(event_times, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact)

    # Step 3: Initialize asset price and volatility arrays.
    dt = T/num_steps
    drift_term = [0,0]
    covariance_matrix = np.array([[1,rho], [rho,1]])
    
    asset_price = np.full(shape=(num_steps+1,num_sims), fill_value=S0)
    asset_volatility = np.full(shape=(num_steps+1,num_sims), fill_value=v0)
    
    Z = np.random.multivariate_normal(drift_term, covariance_matrix, (num_steps,num_sims))

    current_event_index = 0
    next_event_time = event_times[current_event_index] if event_times else None
    
    #4. Simulate asset prices and volatility for each time step and scenario, incorporating event impacts when necessary.
    for i in tqdm(range(1, num_steps + 1), desc="Simulation progress", ncols=100):
        current_time = i * dt

        asset_price[i] = asset_price[i - 1] * np.exp((r - 0.5 * asset_volatility[i - 1]) * dt + np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 0])
        asset_volatility[i] = np.maximum(asset_volatility[i - 1] + kappa * (theta - asset_volatility[i - 1]) * dt + sigma * np.sqrt(asset_volatility[i - 1] * dt) * Z[i - 1, :, 1], 0)

        while next_event_time is not None and current_time >= next_event_time:
            asset_price[i] *= price_impacts[current_event_index]
            asset_volatility[i] *= volatility_impacts[current_event_index]
            
            current_event_index += 1
            if current_event_index < len(event_times):
                next_event_time = event_times[current_event_index]
            else:
                next_event_time = None

    return asset_price, asset_volatility, event_times, price_impacts, volatility_impacts

def simulation_and_plot_hqh_osterlee():
    # Parameters for Q-Hawkes Process
    mu = 1  # Baseline intensity for Hawkes process
    alpha = 0.8  # Positive constant affecting the intensity of the excitation for Hawkes process
    beta = 1.5  # Positive constant affecting the decay rate of the excitation for Hawkes process
    T = 20  # Total runtime given in years
    dt = T / 1000  # Time step length. Should evenly divide T

    # Call Q-Hawkes process
    event_times, memory_loss_times, event_intensities, memory_loss_intensities, Q_values = q_hawkes_process_osterlee(mu, alpha, beta, T, dt)

    # Parameters for Heston model
    S0 = 100.0  # Initial asset price
    v0 = 0.25**2  # Initial variance
    rho = 0.7  # Correlation between asset returns and variance
    kappa = 3  # Rate of mean reversion in variance process
    theta = 0.20**2  # Long-term mean of variance process
    sigma = 0.6  # Volatility of volatility, degree of randomness in the variance process
    num_steps = 1000  # Number of time steps
    num_sims = 1  # Number of scenarios/simulations
    r = 0.02  # Risk-free interest rate
    mean_price_impact = 1.002  # Mean of the normal distribution for price impacts
    std_price_impact = 0.05  # Standard deviation of the normal distribution for price impacts
    mean_vol_impact = 1.002  # Mean of the normal distribution for volatility impacts
    std_vol_impact = 0.05  # Standard deviation of the normal distribution for volatility impacts

    # Run Heston model simulation
    asset_price, asset_volatility, event_times, price_impacts, volatility_impacts = heston_queue_hawkes_normal_osterlee(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact, event_times)

    # Create time array for plotting
    time_array = np.linspace(0, T, num_steps+1)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot asset price
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Asset Price', color=color)
    ax1.plot(time_array, asset_price, color=color, label='Asset Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(np.mean(asset_price), color=color, linestyle='dotted')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'

    # We already handled the x-label with ax1
    ax2.set_ylabel('Volatility', color=color)
    ax2.plot(time_array, asset_volatility, color=color, label='Volatility')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(np.mean(asset_volatility), color=color, linestyle='dotted')

    # Plot event times
    ax1.scatter(event_times, np.interp(event_times, time_array, asset_price[:, 0]), color='purple', alpha=0.5, label='Event times')
    ax2.scatter(event_times, np.interp(event_times, time_array, asset_volatility[:, 0]), color='purple', alpha=0.5)

    # Add title and legend
    fig.suptitle('Asset Price and Volatility Simulation with Q-Hawkes Process')
    fig.legend(loc="upper left")

    # Show plot
    plt.show()

def hqh_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact, mu, alpha, beta, dt, strike, option_type):
    # Run the HQH simulation
    event_times, memory_loss_times, event_intensities, memory_loss_intensities, Q_values = q_hawkes_process_osterlee(mu, alpha, beta, T, dt)
    asset_price, asset_volatility, event_times, price_impacts, volatility_impacts = heston_queue_hawkes_normal_osterlee(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact, event_times)

    # Calculate the terminal asset price
    ST = asset_price[-1]

    # Calculate the payoff for call and put options
    if option_type == 'call':
        payoff = max(ST - strike, 0)
    elif option_type == 'put':
        payoff = max(strike - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount the payoff back to today
    option_price = np.exp(-r * T) * payoff
    return option_price

def hqh_option_example():
    # Fixed parameters for the HQH model
    S0 = 100.0  # Initial asset price
    v0 = 0.25**2  # Initial variance
    rho = 0.7  # Correlation between asset returns and variance
    kappa = 3  # Rate of mean reversion in variance process
    theta = 0.20**2  # Long-term mean of variance process
    sigma = 0.6  # Volatility of volatility, degree of randomness in the variance process
    T = 1  # Option expiry time (in years)
    num_steps = 1000  # Number of time steps
    num_sims = 1  # Number of scenarios/simulations
    r = 0.02  # Risk-free interest rate
    mean_price_impact = 1.002  # Mean of the normal distribution for price impacts
    std_price_impact = 0.05  # Standard deviation of the normal distribution for price impacts
    mean_vol_impact = 1.002  # Mean of the normal distribution for volatility impacts
    std_vol_impact = 0.05  # Standard deviation of the normal distribution for volatility impacts
    mu = 1  # Baseline intensity for Hawkes process
    alpha = 0.8  # Positive constant affecting the intensity of the excitation for Hawkes process
    beta = 1.5  # Positive constant affecting the decay rate of the excitation for Hawkes process
    dt = T / num_steps  # Time step length. Should evenly divide T

    # Range of strike prices
    strikes = np.linspace(80, 120, 50)

    # Calculate the call option price for each strike price
    call_prices = [hqh_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact, mu, alpha, beta, dt, strike, 'call') for strike in strikes]



    # Plot the call option prices
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, call_prices, label='Call Option Price')
    plt.title('Call Option Price vs. Strike Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.legend()
    plt.show()





################################################## Illustrative functions

def illustrate_heston():
    # Illustrate_heston: A wrapper function for running multiple illustrative examples of the Heston model
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
    heston_model_MonteCarlo_results = heston_model_MonteCarlo(S0, v0, rho, kappa, theta, sigma,T, num_steps, num_sims, r)
    simulated_stock_price = heston_model_MonteCarlo_results["asset_prices"]
    simulated_volatility = heston_model_MonteCarlo_results["volatility"]
    
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
    # Single_run_heston_volatility_vs_price: Simulates asset prices and variances over time using the Heston model for a single run with user-specified parameters
    """
    Simulates asset prices and variances over time using the Heston model for a single run with user-specified parameters.
    """
    num_sims = 1

    # Prompt user for Heston parameter values
    auto = input("Do you want to enter your own parameter values? (y/n, default is 'n') ").lower()

    if auto == "y":
        while True:
            S0 = float(input(f"Enter initial asset price (typical range: varies by asset): "))
            T = float(input(f"Enter time horizon in years (typical range: 0 < T < 10): "))
            r = float(input(f"Enter risk-free interest rate (typical range: 0 < r < 0.1): "))
            num_steps = int(input(f"Enter number of time steps in simulation (typical range: 10 < num_steps < 1000): "))
            kappa = float(input(f"Enter rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10): "))
            theta = float(input(f"Enter long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1) (most typical value: {0.20**2:.2f}): "))
            v0 = float(input(f"Enter initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1): "))
            rho = float(input(f"Enter correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1): "))
            sigma = float(input(f"Enter volatility of volatility (typical range: 0 < sigma < 1): "))
            
            if heston_parameters_are_valid(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r):
                print("")
            else:
                user_input = input("Bad input detected. 'c' = continue despite abnormal input. 'r' = retry with new input: ")

            if user_input.lower() == 'r':
                print("")
            else:
                break
    else:
        # Use example Heston parameter values
        S0 = 100.0             # initial asset price (typical range: varies by asset)
        T = 1.0                # time horizon in years (typical range: 0 < T < 10)
        r = 0.02               # risk-free interest rate (typical range: 0 < r < 0.1)
        num_steps = 252        # number of time steps in simulation (typical range: 10 < num_steps < 1000)
        kappa = 3              # rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10)
        theta = 0.20**2        # long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1)
        v0 = 0.25**2           # initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1)
        rho = 0.7              # correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1) "instantanious corelation" enligt erik btw W and W_s
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
    correlation__coefficient_btw_volatility_price = calculate_correlation(simulated_stock_price, simulated_volatility)

    # plot asset prices and variances over time
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.set_title(f'Heston Model Asset Prices and Variance (in this run correlation was: {correlation__coefficient_btw_volatility_price:.2f})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price', color='r')
    ax1.plot(np.linspace(0,T,num_steps+1), simulated_stock_price, color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volatility', color='b')
    ax2.plot(np.linspace(0,T,num_steps+1), simulated_volatility, color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    plt.show()

def heston_mc_option_example_run():
    # Heston_mc_example_run: Demonstrates the Heston model Monte Carlo simulation and displays the asset prices and variances
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
    # Heston_mc_number_of_timesteps_convergence: Illustrates the convergence of the Heston model Monte Carlo simulation with respect to the number of time steps
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
    num_sims = 50000   # Number of Monte Carlo simulations (typical range: 10 < num_sims < 1000)
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
    # Error vs sims kolla bara 4 punkter och större span
    """    
    This function plots:
    - The error of the Heston_MC method v.s. number of simulations
    - Linear best-fit approximation to the error on a log-log scale
    """
    print("Calculating Heston value convergence, this may take a while...")

    S0 = 100.0         # Initial stock price (depends on asset)
    v0 = 0.05          # Initial variance (typical range: 0 < v0 < 1)
    rho = -0.5         # Correlation between stock returns and variance (typical range: -1 < rho < 1)
    kappa = 2.0        # Mean reversion rate of the variance process (typical range: 0 < kappa < 10)
    theta = 0.05       # Long-term mean of the variance process (typical range: 0 < theta < 1)
    sigma = 1.0        # Volatility of the variance process (volatility of volatility) (typical range: 0 < sigma < 1)
    T = 1.0            # Time to expiration (in years) (typical range: 0 < T < 10)
    num_steps = 1000    # Number of time steps in simulation (typical range: 10 < num_steps < 1000)
    r = 0.02           # Risk-free interest rate (typical range: 0 < r < 0.1)
    option_type = 'call' # Type of the option ('call' or 'put')
    K = 110.0          # Strike price of the option
    good_approx_sims = 100000 # Number of simulations for the "good" approximation (typically large)
    ammmount_of_simulation_steps = 20     # Maximum number of simulations to consider (typically in the range of 10 to 1000)

    # Calculate the "good" approximation using a large number of simulations
    good_approx = heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, good_approx_sims, r, option_type, K)
    print("Reference approximation calculated")
    
    # Initialize arrays to store Heston option prices and errors for different numbers of simulations
    heston_values_vs_sims = np.zeros(ammmount_of_simulation_steps)
    heston_mc_err_vs_sims = np.zeros(ammmount_of_simulation_steps)
    
    # Loop over different numbers of simulations and calculate Heston option prices
    num_sims = np.logspace(1, 4, num=ammmount_of_simulation_steps, dtype=int)
    total = len(num_sims)
    for i, sims in enumerate(num_sims):
        heston_values_vs_sims[i] = heston_option(S0, v0, rho, kappa, theta, sigma, T, num_steps, sims, r, option_type, K)
        heston_mc_err_vs_sims[i] = np.abs(good_approx - heston_values_vs_sims[i])
        print(f'{(i+1)/total:.2%} simulations complete')

    
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
    # ℹ️ heston_Volatility: Simulates volatilities using the Heston model and displays the volatilities over time
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
    # ℹ️ illustrate_heston_Volatility: Wrapper function for running the heston_Volatility function to demonstrate volatilities using the Heston model
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

def illustrate_hawkes():
    """
    Plot a Hawkes process with an exponential excitation function.
    Parameters:
    """
    mu = 1 # baseline intensity
    alpha = 5.8 # positive constant affecting the intensity of the excitation
    beta = 1.5 # positive constant affecting the decay rate of the excitation
    T = 10 #time period to simulate the process

    events = hawkes_process(mu, alpha, beta, T)
    event_times = np.array(events)
    time = np.linspace(0, T, 1000)
    intensity = mu + np.array([np.sum(exponential_excitation(t - event_times[event_times < t], alpha, beta)) for t in time])
    
    event_indices = [np.argmin(np.abs(time - et)) for et in event_times]


    print("Printing out the plotted values:")
    print("Time\tIntensity\tEvent")
    for i, t_value in enumerate(time):
        intensity_value = intensity[i]
        event_value = "Event" if i in event_indices else ""
        print(f"{t_value:.2f}\t{intensity_value:.2f}\t{event_value}")

    plt.figure(figsize=(10, 6))
    plt.plot(time, intensity, label="Intensity")
    plt.stem(event_times, [mu]*len(event_times), linefmt="C1-", markerfmt="C1o", basefmt="C1-", label="Events")
    plt.xlabel("Time")
    plt.ylabel("Intensity")
    plt.title("Hawkes Process with Exponential Excitation")
    plt.legend()
    plt.show()

def plot_volatility_smile():
    # Simulate some example data for the volatility smile
    strike_prices = np.linspace(50, 150, 100)
    implied_volatility = (strike_prices - 100)**2 / 10000 + 0.1

    # Plot the volatility smile
    fig, ax = plt.subplots()
    ax.plot(strike_prices, implied_volatility, label='Volatility Smile')
    
    # Add a red vertical line at the current asset price (100)
    ax.axvline(x=100, color='r', linestyle='--', label='Current Asset Price')
    ax.legend()

    # Set the labels and title for the primary x-axis
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('Volatility Smile')

    # Create a secondary x-axis to show the delta between the current asset price and the strike price
    ax2 = ax.twiny()
    delta = strike_prices - 100
    ax2.plot(delta, np.zeros_like(delta), alpha=0)  # Create an invisible plot to synchronize the two x-axes
    ax2.set_xlabel('Delta from Current Asset Price')
    
    # Display the plot
    plt.show()

def plot_heston_hawkes_volatility_and_price():
    num_sims = 1 # Here we only look at one run of price and volatility to clearly see the one example path

    user_wants_to_enter_their_own_parameter_values = input("Do you want to enter your own parameter values? (y/n, default is 'n') ").lower()

    if user_wants_to_enter_their_own_parameter_values == "y":
        while True:
            S0 = float(input(f"Enter initial asset price (typical range: varies by asset): "))
            T = float(input(f"Enter time horizon in years (typical range: 0 < T < 10): "))
            r = float(input(f"Enter risk-free interest rate (typical range: 0 < r < 0.1): "))
            num_steps = int(input(f"Enter number of time steps in simulation (typical range: 10 < num_steps < 1000): "))
            kappa = float(input(f"Enter rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10): "))
            theta = float(input(f"Enter long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1) (most typical value: {0.20**2:.2f}): "))
            v0 = float(input(f"Enter initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1): "))
            rho = float(input(f"Enter correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1): "))
            sigma = float(input(f"Enter volatility of volatility (typical range: 0 < sigma < 1): "))
            hawkes_mu = float(input(f"Enter Hawkes background intensity (mu) (typical range: mu > 0): "))
            hawkes_alpha = float(input(f"Enter Hawkes influence of past events (alpha) (typical range: alpha ≥ 0): "))
            hawkes_beta = float(input(f"Enter Hawkes rate of decay of influence (beta) (typical range: beta > 0): "))

            
            if heston_parameters_are_valid(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r):
                print("")
            else:
                user_input = input("Bad input detected. 'c' = continue despite abnormal input. 'r' = retry with new input: ")

            if user_input.lower() == 'r':
                print("")
            else:
                break
    else:
        # Use example Heston parameter values
        S0 = 100.0             # initial asset price (typical range: varies by asset)
        T = 10                # time horizon in years (typical range: 0 < T < 10)
        r = 0.02               # risk-free interest rate (typical range: 0 < r < 0.1)
        num_steps = 1000       # number of time steps in simulation (typical range: 10 < num_steps < 1000)
        kappa = 3              # rate of mean reversion of variance under risk-neutral dynamics (typical range: 0 < kappa < 10)
        theta = 0.20**2        # long-term mean of variance under risk-neutral dynamics (typical range: 0 < theta < 1)
        v0 = 0.25**2           # initial variance under risk-neutral dynamics (typical range: 0 < v0 < 1)
        rho = 0.7              # correlation between returns and variances under risk-neutral dynamics (typical range: -1 < rho < 1) "instantanious corelation" enligt erik btw W and W_s
        sigma = 0.6            # volatility of volatility (typical range: 0 < sigma < 1)

        # Use example Hawkes parameter values
        hawkes_mu = 1
        hawkes_alpha = 0.8
        hawkes_beta = 1.5

    simulated_stock_price, simulated_volatility, event_times = heston_hawkes(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta)
    correlation__coefficient_btw_volatility_price = calculate_correlation(simulated_stock_price, simulated_volatility)

    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.set_title(f'Heston-Hawkes Model Asset Prices and Volatility (in this run correlation was: {correlation__coefficient_btw_volatility_price:.2f})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price', color='r')
    ax1.plot(np.linspace(0, T, num_steps + 1), simulated_stock_price, color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax1.annotate(
    '(Horizontal purple lines indicate an event occurring)',
    xy=(0.5, -0.1),  # Adjust the y-coordinate for vertical alignment
    xycoords='axes fraction',
    fontsize=10,
    color='purple',
    ha='center',
    va='top',
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel('Volatility', color='b')
    ax2.plot(np.linspace(0, T, num_steps + 1), simulated_volatility, color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Plot the event times as horizontal purple dotted lines
    for event_time in event_times:
        ax1.axvline(event_time, color='purple', linestyle=(0, (3, 3)))

    plt.show()

def illustrate_heston_hawkes_normal_mc():
    # Use example parameter values
    S0 = 100.0
    v0 = 0.25**2
    rho = 0.7
    kappa = 3
    theta = 0.20**2
    sigma = 0.6
    T = 20
    num_steps = 1000
    num_sims = 1
    r = 0.02
    hawkes_mu = 1
    hawkes_alpha = 0.8
    hawkes_beta = 1.5
    mean_price_impact = 1.0
    std_price_impact = 0.1
    mean_vol_impact = 1.0
    std_vol_impact = 0.1

    asset_price, asset_volatility, event_times, price_impacts, volatility_impacts = heston_hawkes_normal_mc(S0, v0, rho, kappa, theta, sigma, T, num_steps, num_sims, r, hawkes_mu, hawkes_alpha, hawkes_beta, mean_price_impact, std_price_impact, mean_vol_impact, std_vol_impact)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_title('Heston-Hawkes-Normal Model Asset Prices and Volatility')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price', color='r')
    ax1.plot(np.linspace(0, T, num_steps + 1), asset_price, color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax1.annotate(
        '(Horizontal purple lines indicate an event occurring)',
        xy=(0.5, -0.1),
        xycoords='axes fraction',
        fontsize=10,
        color='purple',
        ha='center',
        va='top',
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel('Volatility', color='b')
    ax2.plot(np.linspace(0, T, num_steps + 1), asset_volatility, color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Plot the event times as horizontal purple dotted lines
    for event_time in event_times:
        ax1.axvline(event_time, color='purple', linestyle=(0, (3, 3)))

    plt.show()

    # Print the mean and standard deviation of the normal distributions
    print("\n\nAn impact of 1 would mean mutiplying the number by 1 and thus no impact. 1.1 woul mean increasing the value by 10 percent. \nFollowing are the impact values drawn from the normal distribution of both volatility and price: \n")
    print(f"Mean Price Impact: {mean_price_impact:.2f}, Std Dev Price Impact: {std_price_impact:.2f}")
    print(f"Mean Volatility Impact: {mean_vol_impact:.2f}, Std Dev Volatility Impact: {std_vol_impact:.2f}\n")

    # Print the table header
    print("Event Time\tPrice Impact (%)\tVolatility Impact (%)")

    # Print the table rows
    for event_time, price_impact, vol_impact in zip(event_times, price_impacts, volatility_impacts):
        print(f"{event_time:.2f}\t\t{(price_impact - 1) * 100:.2f}\t\t\t{(vol_impact - 1) * 100:.2f}")

    print("\n\n\n")


def main():
    # Main: The main function that executes the program and calls the appropriate functions based on user input
    print("Choose test")
    print("1: 'illustrate_heston' with example values")
    print("2: Single run of Heston showing stockprice and volatility")
    print("3: Make one example run of Heston model to calculate an Option price")
    print("4: Plot heston_mc_number_of_timesteps_convergence")
    print("5: Plot heston_mc_number_of_simulations_convergence")
    print("6: Visualize Heston Volatility")
    print("7: illustrate_hawkes() process of self-excitation")
    print("8: plot_volatility_smile() With example values")
    print("9: plot_heston_hawkes_volatility_and_price")
    print("10: illustrate_heston_hawkes_normal_mc(). Example run showing Heston Hawkes with normal distributed price and vol impacts")
    print("11: run_and_plot_q_hawkes(). Plots an example run of a Q-Hawkes process.")
    print("12: illustrate_heston_queue_hawkes_normal_mc() Makes one example run of a HQH process and plots price vs Volatility and events")
    print("13: Makes one example run of the Q-Hawkes process (as implemented by Osterlee)")
    print("14: Make an example run of the HQH method and plots price vs volatility")
    user_choice = int(input("\n------> Choose test: "))

    if user_choice==1:
        illustrate_heston()
    if user_choice==2:
        single_run_heston_volatility_vs_price()
    if user_choice==3:
        heston_mc_option_example_run()
    if user_choice==4:
        heston_mc_number_of_timesteps_convergence()
    if user_choice==5:
        heston_mc_number_of_simulations_convergence()
    if user_choice==6:
        illustrate_heston_Volatility()
    if user_choice==7:
        illustrate_hawkes()
    if user_choice==8:
        plot_volatility_smile()
    if user_choice==9:
        plot_heston_hawkes_volatility_and_price()
    if user_choice==10:
        illustrate_heston_hawkes_normal_mc()
    if user_choice==11:
        run_and_plot_q_hawkes()
    if user_choice==12:
        illustrate_heston_queue_hawkes_normal_mc()
    if user_choice==13:
        plot_q_hawkes_process_osterlee()
    if user_choice==14:
        simulation_and_plot_hqh_osterlee()
    else:
        print("Invalid input into the main program")

    print("\n\nEnd of main program.")

main()