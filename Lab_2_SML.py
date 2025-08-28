#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Example data
data = [
    {'age': 23, 'shear': 2.5},
    {'age': 45, 'shear': 4.1},
    {'age': 31, 'shear': 3.2},
    {'age': 50, 'shear': 4.7},
    {'age': 28, 'shear': 3.0}
]

# Convert list of dicts to DataFrame
df = pd.DataFrame(data)

# Define dependent and independent variables
y = df['shear'].values           # dependent variable
x = df['age'].values             # independent variable

# Add a constant term to the independent variables (for the intercept)
x_const = sm.add_constant(x)

# --- OLS using statsmodels for reference ---
linear_regression = sm.OLS(y, x_const)
fitted_model = linear_regression.fit()
print("Statsmodels OLS summary:")
print(fitted_model.summary())

# --- Gradient Descent Implementation ---

# Initialize parameters
beta_0 = 0.0  # intercept
beta_1 = 0.0  # slope

# Hyperparameters
learning_rate = 0.0001
n_iterations = 10000
n = float(len(y))

# Gradient Descent Loop
for i in range(n_iterations):
    y_pred = beta_0 + beta_1 * x
    error = y_pred - y
    # Calculate gradients
    grad_beta_0 = (2/n) * np.sum(error)
    grad_beta_1 = (2/n) * np.sum(error * x)
    # Update parameters
    beta_0 -= learning_rate * grad_beta_0
    beta_1 -= learning_rate * grad_beta_1

    # Optionally print loss every 1000 iterations
    if i % 1000 == 0:
        mse = (1/n) * np.sum(error ** 2)
        print(f"Iteration {i}: MSE = {mse:.4f}, beta_0 = {beta_0:.4f}, beta_1 = {beta_1:.4f}")

print("\nGradient Descent results:")
print(f"Intercept (beta_0): {beta_0}")
print(f"Slope (beta_1): {beta_1}")

# --- Plotting ---

plt.scatter(x, y, label='Data points', color='blue')

# Predicted values from OLS
y_pred_ols = fitted_model.predict(x_const)
sorted_idx = np.argsort(x)
plt.plot(x[sorted_idx], y_pred_ols[sorted_idx], color='red', label='OLS fitted line')

# Predicted values from Gradient Descent
y_pred_gd = beta_0 + beta_1 * x
plt.plot(x[sorted_idx], y_pred_gd[sorted_idx], color='green', linestyle='--', label='Gradient Descent')

plt.xlabel('Age')
plt.ylabel('Shear')
plt.title('Linear Regression Fit: OLS vs Gradient Descent')
plt.legend()
plt.show()


# In[ ]:




