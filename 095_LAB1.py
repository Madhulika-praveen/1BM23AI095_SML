#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Example data: list of dictionaries (replace this with your actual data)
data = [
    {'age': 23, 'shear': 2.5},
    {'age': 45, 'shear': 4.1},
    {'age': 31, 'shear': 3.2},
    {'age': 50, 'shear': 4.7},
    {'age': 28, 'shear': 3.0}
]

# Convert list of dicts to DataFrame
df = pd.DataFrame(data)

# Print the DataFrame head to check
print(df.head())

# Define dependent and independent variables
y = df['shear']           # dependent variable
x = df['age']             # independent variable

# Add a constant term to the independent variables (for the intercept)
x_const = sm.add_constant(x)

# Create the OLS model
linear_regression = sm.OLS(y, x_const)

# Fit the model
fitted_model = linear_regression.fit()

# Print the summary of regression results
print(fitted_model.summary())

# Plotting
plt.scatter(x, y, label='Data points', color='blue')

# Predicted values
y_pred = fitted_model.predict(x_const)

# Sort x and y_pred for a clean line plot
sorted_idx = np.argsort(x)
plt.plot(x[sorted_idx], y_pred[sorted_idx], color='red', label='Fitted line')

plt.xlabel('Age')
plt.ylabel('Shear')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()


# In[ ]:




