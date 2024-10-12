import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo simulation for stock prices using Geometric Brownian Motion
def monte_carlo_simulation(S0, mu, sigma, days, simulations):
    """
    Perform Monte Carlo simulation for stock prices.
    
    Parameters:
    - S0: Initial stock price.
    - mu: Expected return (drift).
    - sigma: Volatility (standard deviation).
    - days: Number of days to simulate.
    - simulations: Number of simulation paths.
    
    Returns:
    - A 2D array of stock prices for each simulation.
    """
    
    # Time increment (1 day)
    dt = 1 / 252  # Assuming 252 trading days in a year
    
    # Array to store the results of simulations
    prices = np.zeros((days, simulations))
    
    # Initialize the first day with the initial stock price
    prices[0] = S0
    
    # Simulate stock prices for each day and each simulation
    for t in range(1, days):
        # Generate random numbers from a standard normal distribution
        random_shocks = np.random.normal(0, 1, simulations)
        
        # Apply Geometric Brownian Motion formula to calculate the stock price
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * random_shocks * np.sqrt(dt))
    
    return prices

# Parameters for the simulation
S0 = 100  # Initial stock price
mu = 0.1  # Expected annual return (drift)
sigma = 0.2  # Annual volatility
days = 252  # Number of days to simulate (1 year)
simulations = 1000  # Number of simulation paths

# Run the simulation
simulated_prices = monte_carlo_simulation(S0, mu, sigma, days, simulations)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(simulated_prices[:, :10])  # Plotting the first 10 simulations
plt.title('Monte Carlo Simulation of Stock Prices')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.show()

# Optional: Calculate the mean and final price distribution
mean_final_price = np.mean(simulated_prices[-1])
std_final_price = np.std(simulated_prices[-1])

print(f"Mean final price after {days} days: ${mean_final_price:.2f}")
print(f"Standard deviation of final prices: ${std_final_price:.2f}")
