#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


# In[2]:


# Read the dataset
data = pd.read_csv("C:\\Users\\91762\\Desktop\\Mini project\\data sets\\Rain fall data from 1901 to 2022.csv") 
data.head()


# In[3]:


# Drop unnecessary columns (if needed)
# Adjust this based on the columns you want to include in your features (X)
X = data.drop(['SUBDIVISION', 'Quality'], axis=1)


# In[4]:


# Use 'ANNUAL' as the target variable (rainfall to be predicted)
y = data['ANNUAL']


# In[5]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# Define and train the DBN using BernoulliRBM
rbm = BernoulliRBM(n_components=256, learning_rate=0.1, n_iter=10, random_state=42, verbose=True)


# In[8]:


# Build the pipeline
dbn_pipeline = Pipeline([('rbm', rbm)])


# In[9]:


# Train the DBN
X_train_dbn = dbn_pipeline.fit_transform(X_train_scaled)


# In[10]:


# Optionally, you can train a regression model on top of the DBN features
from sklearn.linear_model import LinearRegression


# In[11]:


# Check for missing values in X_train_dbn
missing_values = np.isnan(X_train_dbn).any()
print(f'Missing values in X_train_dbn: {missing_values}')


# In[12]:


# If there are missing values, you can handle them by filling with the mean
X_train_dbn = np.nan_to_num(X_train_dbn, nan=np.nanmean(X_train_dbn))


# In[13]:


print(data.columns)


# In[14]:


regression_model = LinearRegression()
regression_model.fit(X_train_dbn, y_train)


# In[15]:


# Make predictions on the test set
X_test_dbn = dbn_pipeline.transform(X_test_scaled)
y_pred = regression_model.predict(X_test_dbn)


# In[16]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[17]:


import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
# Replace 'x_column' and 'y_column' with the names of the columns you want to plot
x_column = 'ANNUAL'  # Replace with your actual column name
y_column = 'Maximum temperature (Degree C)'  # Replace with your actual column name

# Scatter plot
plt.scatter(data[x_column], data[y_column], color='blue', alpha=0.5)
plt.title(f'Scatter Plot of {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()


# In[18]:


import seaborn as sns

# Assuming 'df' is your DataFrame
# Replace 'x_column' and 'y_column' with the names of the columns you want to plot
x_column = 'ANNUAL'  # Replace with your actual column name
y_column = 'Minimum temperature (Degree C)'  # Replace with your actual column name

# Scatter plot with Seaborn
sns.scatterplot(x=data[x_column], y=data[y_column], color='blue', alpha=0.5)
plt.title(f'Scatter Plot of {x_column} vs {y_column}')
plt.show()


# In[20]:


import seaborn as sns
# Pair plot for selected columns
sns.pairplot(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']])
plt.suptitle('Pair Plot for Monthly Rainfall')
plt.show()


# In[ ]:




