import numpy as np
import matplotlib.pyplot as plt

# 1. Setup the data points
area = np.array([1000, 1001])
price = np.array([200000, 250000])

# 2. Parameters for Regularization
lambda_val = 0.1  # The "Tax" strength
bias = 0          # Simplified bias for visualization

# 3. Create a range of possible weights to test
weights = np.linspace(0, 55000, 500)

# 4. Calculate Costs
mse_costs = []
penalty_costs = []
total_costs = []

for w in weights:
    # Prediction Error (Mean Squared Error)
    predictions = w * area + bias
    mse = np.mean((predictions - price)**2)
    
    # Ridge Penalty (lambda * w^2)
    penalty = lambda_val * (w**2)
    
    mse_costs.append(mse)
    penalty_costs.append(penalty)
    total_costs.append(mse + penalty)

# 5. Find the "Sweet Spot" (Minimum Total Cost)
optimal_w = weights[np.argmin(total_costs)]

# 6. Plotting
plt.figure(figsize=(10, 6))
plt.plot(weights, mse_costs, label='Prediction Error (MSE)', color='red', linestyle='--')
plt.plot(weights, penalty_costs, label='Weight Penalty (Tax)', color='blue', linestyle='--')
plt.plot(weights, total_costs, label='TOTAL COST (Error + Penalty)', color='black', linewidth=2)

# Mark the points
plt.axvline(x=50000, color='gray', alpha=0.5, label='Original Overfitted Weight')
plt.axvline(x=optimal_w, color='green', linewidth=2, label=f'Regularized Weight (~{int(optimal_w)})')

plt.title('How Regularization Pulls the Weight Down')
plt.xlabel('Weight (w)')
plt.ylabel('Cost Value')
# plt.legend()
plt.grid(True, alpha=0.3)
plt.show()