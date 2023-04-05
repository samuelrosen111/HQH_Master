import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
from scipy.stats import poisson
import matplotlib.pyplot as plt


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
    
    while t < T:
        lambda_max = mu + alpha  # Upper bound on intensity (assuming exponential excitation)
        t += -np.log(np.random.rand()) / lambda_max  # Generate inter-event time from an exponential distribution
        
        if t >= T:
            break
        
        # Acceptance-rejection sampling
        p = mu / lambda_max
        if len(events) > 0:
            p += exponential_excitation(t - events[-1], alpha, beta) / lambda_max
        
        if np.random.rand() < p:
            events.append(t)
    
    return events

def illustrate_hawkes():
    """
    Plot a Hawkes process with an exponential excitation function.
    Parameters:
    """
    mu = 1 # baseline intensity
    alpha = 0.8 # positive constant affecting the intensity of the excitation
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

################################################################################

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

plot_volatility_smile()

################################################################################ 1

def q_hawkes(mu, alpha, beta, T):
    t = 0
    events = []
    queue = []

    while t < T:
        if not queue:
            tau = np.random.exponential(1 / mu)
            t += tau
            events.append(t)
            queue.append(np.random.exponential(1 / beta))
        else:
            tau_min = min(queue)
            t += tau_min
            idx_min = queue.index(tau_min)

            if idx_min == 0:
                events.append(t)
                tau = np.random.exponential(1 / (alpha * mu))
                queue.append(tau)
            else:
                queue[idx_min] -= tau_min

            queue = [tau - tau_min for tau in queue]
            queue.pop(idx_min)

    return events

def plot_q_hawkes():
    # Declare example parameters
    mu = 0.2     # Base event rate
    alpha = 0.5  # Excitation parameter
    beta = 1.0   # Decay parameter
    T = 100      # Total simulation time

    # Call the q_hawkes function with the example parameters
    events = q_hawkes(mu, alpha, beta, T)

    # Plot the events
    plt.figure(figsize=(10, 3))
    plt.plot(events, np.ones_like(events), 'bo', ms=8, label='Events')
    plt.xlim(0, T)
    plt.yticks([])
    plt.xlabel('Time')
    plt.title('Queue-Hawkes Process')
    plt.legend(loc='upper right')
    plt.show()

#plot_q_hawkes()

################################################################################ 2

def simulate_queue_hawkes_process(base_intensity, excitation, decay_rate, num_events):
    arrival_times = []
    t = 0
    while len(arrival_times) < num_events:
        if len(arrival_times) == 0:
            rate = base_intensity
        else:
            rate = base_intensity + np.sum(excitation * np.exp(-decay_rate * (t - np.array(arrival_times))))
        t += np.random.exponential(scale=1/rate)
        arrival_times.append(t)
    return arrival_times

def plot_queue_hawkes_process():
    base_intensity = 0.1
    excitation = 0.5
    decay_rate = 1.0
    num_events = 100

    arrival_times = simulate_queue_hawkes_process(base_intensity, excitation, decay_rate, num_events)

    plt.plot(arrival_times, np.arange(num_events))
    plt.xlabel('Time')
    plt.ylabel('Number of events')
    plt.title('Queue-Hawkes Process')
    plt.show()

#plot_queue_hawkes_process()



def main():
    print("Choose test")
    print("1: 'illustrate_hawkes' with example values")
    user_choice = int(input("\n------> Choose test: "))
    if user_choice==1:
        illustrate_hawkes()

# Experiment

def heston_characteristic_function(u, t, r, V0, theta, kappa, sigma, rho):
    """Returns the characteristic function of the Heston model."""
    a = kappa * theta
    b = kappa + rho * sigma * u * 1j
    c = sigma ** 2 / 2

    D = np.sqrt(b ** 2 - 4 * a * c)

    r_plus = (b + D) / (2 * c)
    r_minus = (b - D) / (2 * c)

    g = r_minus / r_plus

    C = r * u * t * 1j

    return np.exp(C + V0 / sigma ** 2 * ((b - D) * t / (2 * c) - 2 * np.log((1 - g * np.exp(-D * t)) / (1 - g))))

def Q_Hawkes_characteristic_function(u, t, r, V0, theta, kappa, sigma, alpha, beta):
    """Returns the characteristic function of the Q-Hawkes model."""
    c = alpha / (beta + alpha * t)

    return np.exp(1j * u * (r * t + V0 * c)) * np.exp(-theta * t * (1 - c) * (1 - np.exp(1j * u * alpha / sigma ** 2))) ** (-kappa / theta)

def HQH_characteristic_function(u, t, r, S0, V0, theta, kappa, sigma, rho, alpha, beta):
    """Returns the characteristic function of the HQH model."""
    Heston_CF = heston_characteristic_function(u - 1j, t, r, V0, theta, kappa, sigma, rho)
    Q_Hawkes_CF = Q_Hawkes_characteristic_function(u, t, r, V0, theta, kappa, sigma, alpha, beta)

    return np.exp(1j * u * np.log(S0) + Heston_CF + Q_Hawkes_CF)

def COS_method(N, S, K, r, T, alpha, sigma, rho, kappa, theta, lambda_):
    """
    Extension of the COS method to cope with discrete distributions
    for pricing European and Bermudan options with HQH jump-diffusion process
    
    Args:
    N: number of points in COS grid
    S: spot price
    K: strike price
    r: risk-free rate
    T: time to maturity
    alpha, sigma, rho, kappa, theta, lambda_: HQH model parameters
    
    Returns:
    c: European option price
    """
    x0 = np.log(S)
    k = np.arange(N)
    b = 2*np.pi*k/(N-1)
    u = b*rho/alpha - sigma/alpha
    a = -0.5*(u**2 + sigma**2)/kappa
    b = kappa*rho*u/alpha + theta/sigma
    q = 0.5*(rho**2 + 1)/alpha
    
    psi = lambda v: np.exp(1j*u*x0/v)*np.exp((a*v**2 + b*v)*T)
    char = lambda v: psi(v)/((1j*v)*psi(-v))
    
    I = np.sum(np.exp(-r*T)*char(b+1j*b**2)*np.cos(b*x0)*np.diff(b))
    c = np.maximum(0, S*np.exp(-lambda_*T)*(I.real + 0.5*(psi(alpha-1j)-I).real))
    
    return c

def main():
    # HQH model parameters
    alpha = 0.1
    sigma = 0.3
    rho = -0.6
    kappa = 1
    theta = 0.04
    lambda_ = 0.5
    
    # Option parameters
    S = 100
    K = 100
    r = 0.05
    T = 1
    
    # Grid parameters
    N = 2**10
    
    # Compute European option price
    c = COS_method(N, S, K, r, T, alpha, sigma, rho, kappa, theta, lambda_)
    print(f"European option price: {c}")
    
#main()

#if __name__ == '__main__':
#  main()
