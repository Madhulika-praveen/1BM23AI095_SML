#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Example data
shear = np.array([2160.70, 1680.15, 2318.00, 2063.30, 2209.30, 2209.50, 1710.30, 1786.70,
                  2577.90, 2359.90, 2258.70, 2167.20, 2401.55, 1781.80, 2338.75, 1767.30,
                  2055.50, 2416.40, 2202.50, 2656.20, 1755.70])
age = np.array([15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50, 7.50, 11.00, 13.00,
                3.75, 25.00, 9.75, 22.00, 18.00, 6.00, 12.50, 2.00, 21.50, 0.00])

n = len(shear)

# Initialize parameters
beta_0 = 0.0  # intercept
beta_1 = 0.0  # slope

# Hyperparameters
learning_rate = 0.0001
epochs = 50

# SGD loop
for epoch in range(epochs):
    for i in range(n):
        xi = age[i]
        yi = shear[i]

        # Prediction for ith data point
        y_pred_i = beta_0 + beta_1 * xi

        # Calculate error
        error_i = y_pred_i - yi

        # Compute gradients
        grad_beta_0 = 2 * error_i
        grad_beta_1 = 2 * error_i * xi

        # Update parameters
        beta_0 -= learning_rate * grad_beta_0
        beta_1 -= learning_rate * grad_beta_1

    # Calculate and print MSE after each epoch
    y_pred = beta_0 + beta_1 * age
    mse = (1/n) * np.sum((y_pred - shear) ** 2)
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse:.4f}, beta_0: {beta_0:.4f}, beta_1: {beta_1:.4f}")

# Plotting results
plt.scatter(age, shear, color='blue', label='Data points')
plt.plot(np.sort(age), beta_0 + beta_1 * np.sort(age), color='orange', linestyle='-', label='SGD Regression Line')
plt.xlabel('Age')
plt.ylabel('Shear')
plt.title('Linear Regression via Stochastic Gradient Descent')
plt.legend()
plt.show()

