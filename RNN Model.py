#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# In[2]:


# Read the dataset
data = pd.read_csv("C:\\Users\\91762\\Desktop\\Mini project\\data sets\\Rain fall data from 1901 to 2022.csv") 
data.head()


# In[3]:


features = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT','NOV', 'DEC']

# Define input features and target variable
X = data[features]
y = data['ANNUAL']


# In[4]:


print(data.columns)


# In[5]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for RNN (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# In[6]:


model_rnn = Sequential()
model_rnn.add(SimpleRNN(units=50, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model_rnn.add(Dense(units=1))

# Compile the model
model_rnn.compile(optimizer='adam', loss='mean_squared_error')


# In[7]:


model_rnn.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))


# In[8]:


# Make predictions using RNN
predictions_rnn = model_rnn.predict(X_test_reshaped)

# Evaluate RNN model
mse_rnn = mean_squared_error(y_test, predictions_rnn)
print(f'Mean Squared Error (RNN): {mse_rnn}')


# In[9]:


plt.plot(y_test.values, label='Actual')
plt.plot(predictions_rnn, label='Predicted (RNN)')
plt.legend()
plt.title('RNN Model Prediction')
plt.show()


# In[10]:


# Scatter plot of actual vs predicted values
plt.scatter(y_test, predictions_rnn)
plt.title('Actual vs Predicted Scatter Plot (RNN)')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.show()


# In[11]:


residuals_rnn = y_test - predictions_rnn.flatten()

plt.scatter(predictions_rnn.flatten(), residuals_rnn)
plt.title('Residual Plot (RNN)')
plt.xlabel('Predicted Rainfall')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()


# In[12]:


plt.hist(residuals_rnn, bins=30)
plt.title('Distribution of Residuals (RNN)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[13]:


plt.plot(y_test, label='Actual')
plt.plot(predictions_rnn, label='Predicted (RNN)')
plt.legend()
plt.title('Actual vs Predicted Rainfall (RNN)')
plt.xlabel('Time Steps')
plt.ylabel('Rainfall')
plt.show()


# In[14]:


import seaborn as sns


# In[15]:


# Pair plot for selected columns
sns.pairplot(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']])
plt.suptitle('Pair Plot for Monthly Rainfall')
plt.show()


# In[16]:


# Make predictions using RNN
predictions_rnn = model_rnn.predict(X_test_reshaped)

# Evaluate RNN model
mse_rnn = mean_squared_error(y_test, predictions_rnn)
print(f'Mean Squared Error (RNN): {mse_rnn}')


# In[17]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math


# In[18]:


# Root Mean Squared Error (RMSE)
rmse_rnn = math.sqrt(mse_rnn)
print(f'Root Mean Squared Error (RNN): {rmse_rnn}')


# In[19]:


# Mean Absolute Error (MAE)
mae_rnn = mean_absolute_error(y_test, predictions_rnn)
print(f'Mean Absolute Error (RNN): {mae_rnn}')


# In[20]:


# R-squared Score
r2_score_rnn = r2_score(y_test, predictions_rnn)
print(f'R-squared Score (RNN): {r2_score_rnn}')


# In[ ]:




