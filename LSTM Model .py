#!/usr/bin/env python
# coding: utf-8

# # Data Preparation and Exploration

# In[1]:


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from keras.models import Sequential
from keras.layers import Dense, LSTM


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


# # Using LSTM Model

# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Use MinMaxScaler to scale data between 0 and 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[8]:


# Reshape data for LSTM input (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


# In[9]:


model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[10]:


model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[11]:


model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))


# In[12]:


# Make predictions on the test set
predictions = model.predict(X_test_reshaped)


# In[13]:


from sklearn.metrics import mean_squared_error

# Assuming 'predictions' are the predicted values from your LSTM model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# In[14]:


rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')


# In[15]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')


# In[16]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
print(f'R-squared Score: {r2}')


# In[17]:


import matplotlib.pyplot as plt

# Visualize predictions
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()


# In[18]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['ANNUAL'], label='Actual')
plt.plot(predictions, label='Predicted', linestyle='dashed')
plt.title('Rainfall Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()


# In[19]:


# Assuming 'mse_values' is a list of MSE values for each prediction step
mse_values = [7284709.405223398,2699.020082404612,745.1209373393029,-0.005796967739083003]

# Visualization of Mean Squared Error Over Time
plt.figure(figsize=(10, 5))
plt.plot(mse_values, marker='o')
plt.title('Mean Squared Error Over Time')
plt.xlabel('Prediction Step')
plt.ylabel('Mean Squared Error')
plt.show()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[21]:


# Pair plot for selected columns
sns.pairplot(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']])
plt.suptitle('Pair Plot for Monthly Rainfall')
plt.show()


# In[22]:


# Distribution plot for a specific column
plt.figure(figsize=(8, 6))
sns.histplot(data['ANNUAL'], kde=True, color='skyblue')
plt.title('Distribution of Annual Rainfall')
plt.xlabel('Annual Rainfall')
plt.ylabel('Frequency')
plt.show()


# In[3]:


pip install model


# In[5]:


import sys
sys.path.append('/path/to/module')

