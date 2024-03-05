#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model


# In[2]:


# Read the dataset
data = pd.read_csv("C:\\Users\\91762\\Desktop\\Mini project\\data sets\\Rain fall data from 1901 to 2022.csv") 
data.head()


# In[3]:


print(data.columns)


# In[4]:


# Extract features (X) and target variable (y)
X = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].values
y = data['ANNUAL'].values


# In[5]:


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# Define the autoencoder model
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(8, activation='relu')(input_layer)
decoded = Dense(X_train.shape[1], activation='linear')(encoded)


# In[8]:


autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')


# In[9]:


# Train the autoencoder
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))


# In[10]:


# Use the trained autoencoder for dimensionality reduction
encoder = Model(input_layer, encoded)
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)


# In[11]:


# Display the architecture of the autoencoder
autoencoder.summary()


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[13]:


# Define a regression model (you can use any other regression model as well)
regression_model = LinearRegression()


# In[14]:


# Fit the regression model on the encoded training data
regression_model.fit(X_train_encoded, y_train)


# In[15]:


# Make predictions on the encoded test data
y_pred = regression_model.predict(X_test_encoded)


# In[16]:


# Evaluate the performance of the regression model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[35]:


# Assuming you have y_new_actual, replace it with your actual labels
# y_new_actual = ...

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter([10.5, 15.2, 20.0], [13.5, 17.2, 21.0], color='blue', label='Actual vs Predicted')
plt.plot([min(10.5, 15.2, 20.0), max(13.5, 17.2, 21.0)], [min(10.5, 15.2, 20.0), max(13.5, 17.2, 21.0)], linestyle='--', color='red', label='Ideal Line')
plt.title('Actual vs Predicted Values on New Encoded Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()


# In[ ]:




