#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Example data
data = {
    "shear": [2160.70, 1680.15, 2318.00, 2063.30, 2209.30, 2209.50, 1710.30, 1786.70,
              2577.90, 2359.90, 2258.70, 2167.20, 2401.55, 1781.80, 2338.75, 1767.30,
              2055.50, 2416.40, 2202.50, 2656.20, 1755.70],
    "age": [15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50, 7.50, 11.00, 13.00,
            3.75, 25.00, 9.75, 22.00, 18.00, 6.00, 12.50, 2.00, 21.50, 0.00]
}

df = pd.DataFrame(data)

# Define dependent and independent variables
y = df['shear'].values
x = df['age'].values
n = float(len(y))

# Add a constant for OLS
x_const = sm.add_constant(x)

# --- OLS using statsmodels for reference ---
linear_regression = sm.OLS(y, x_const)
fitted_model = linear_regression.fit()
print("Statsmodels OLS summary:")
print(fitted_model.summary())

# --- Batch Gradient Descent Implementation ---

beta_0 = 0.0
beta_1 = 0.0

learning_rate = 0.0001
n_iterations = 10000

for i in range(n_iterations):
    y_pred = beta_0 + beta_1 * x
    error = y_pred - y

    grad_beta_0 = (2/n) * np.sum(error)
    grad_beta_1 = (2/n) * np.sum(error * x)

    beta_0 -= learning_rate * grad_beta_0
    beta_1 -= learning_rate * grad_beta_1

    if i % 1000 == 0:
        mse = (1/n) * np.sum(error ** 2)
        print(f"Batch GD Iteration {i}: MSE = {mse:.4f}, beta_0 = {beta_0:.4f}, beta_1 = {beta_1:.4f}")

print("\nBatch Gradient Descent results:")
print(f"Intercept (beta_0): {beta_0}")
print(f"Slope (beta_1): {beta_1}")

# --- Stochastic Gradient Descent Implementation ---

beta_0_sgd = 0.0
beta_1_sgd = 0.0
learning_rate_sgd = 0.0001
epochs = 50

for epoch in range(epochs):
    for i in range(int(n)):
        xi = x[i]
        yi = y[i]

        y_pred_i = beta_0_sgd + beta_1_sgd * xi
        error_i = y_pred_i - yi

        grad_beta_0_sgd = 2 * error_i
        grad_beta_1_sgd = 2 * error_i * xi

        beta_0_sgd -= learning_rate_sgd * grad_beta_0_sgd
        beta_1_sgd -= learning_rate_sgd * grad_beta_1_sgd

    # Calculate total MSE at the end of each epoch for monitoring
    y_pred_epoch = beta_0_sgd + beta_1_sgd * x
    mse_epoch = (1/n) * np.sum((y_pred_epoch - y) ** 2)
    if epoch % 10 == 0:
        print(f"SGD Epoch {epoch}: MSE = {mse_epoch:.4f}, beta_0 = {beta_0_sgd:.4f}, beta_1 = {beta_1_sgd:.4f}")

print("\nStochastic Gradient Descent results:")
print(f"Intercept (beta_0_sgd): {beta_0_sgd}")
print(f"Slope (beta_1_sgd): {beta_1_sgd}")

# --- Plotting ---

plt.scatter(x, y, label='Data points', color='blue')

# OLS line
y_pred_ols = fitted_model.predict(x_const)
sorted_idx = np.argsort(x)
plt.plot(x[sorted_idx], y_pred_ols[sorted_idx], color='red', label='OLS fitted line')

# Batch GD line
y_pred_gd = beta_0 + beta_1 * x
plt.plot(x[sorted_idx], y_pred_gd[sorted_idx], color='green', linestyle='--', label='Batch Gradient Descent')

# SGD line
y_pred_sgd = beta_0_sgd + beta_1_sgd * x
plt.plot(x[sorted_idx], y_pred_sgd[sorted_idx], color='orange', linestyle=':', label='Stochastic Gradient Descent')

plt.xlabel('Age')
plt.ylabel('Shear')
plt.title('Linear Regression Fit: OLS vs Batch GD vs SGD')
plt.legend()
plt.show()


# In[ ]:




