#!/usr/bin/env python
# coding: utf-8

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
import streamlit as st


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


# # LSTM Model

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


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[20]:


# Pair plot for selected columns
sns.pairplot(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN']])
plt.suptitle('Pair Plot for Monthly Rainfall')
plt.show()


# In[21]:


# Distribution plot for a specific column
plt.figure(figsize=(8, 6))
sns.histplot(data['ANNUAL'], kde=True, color='skyblue')
plt.title('Distribution of Annual Rainfall')
plt.xlabel('Annual Rainfall')
plt.ylabel('Frequency')
plt.show()


# # CNN Model

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


# In[23]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for CNN (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# In[24]:


model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(units=50, activation='relu'))
model_cnn.add(Dense(units=1))

# Compile the model
model_cnn.compile(optimizer='adam', loss='mean_squared_error')


# In[25]:


#Trainning CNN
model_cnn.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))


# In[26]:


# Make predictions using CNN
predictions_cnn = model_cnn.predict(X_test_reshaped)

# Evaluate CNN model
mse_cnn = mean_squared_error(y_test, predictions_cnn)
print(f'Mean Squared Error (CNN): {mse_cnn}')


# In[27]:


plt.plot(y_test.values, label='Actual')
plt.plot(predictions_cnn, label='Predicted (CNN)')
plt.legend()
plt.title('CNN Model Prediction')
plt.show()


# In[28]:


# Scatter plot of actual vs predicted values
plt.scatter(y_test, predictions_cnn)
plt.title('Actual vs Predicted Scatter Plot (CNN)')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.show()


# In[29]:


residuals_cnn = y_test - predictions_cnn.flatten()

plt.scatter(predictions_cnn.flatten(), residuals_cnn)
plt.title('Residual Plot (CNN)')
plt.xlabel('Predicted Rainfall')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()


# In[30]:


plt.hist(residuals_cnn, bins=30)
plt.title('Distribution of Residuals (CNN)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# # RNN Model 

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# In[32]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for RNN (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# In[33]:


model_rnn = Sequential()
model_rnn.add(SimpleRNN(units=50, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model_rnn.add(Dense(units=1))

# Compile the model
model_rnn.compile(optimizer='adam', loss='mean_squared_error')


# In[34]:


model_rnn.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))


# In[35]:


# Make predictions using RNN
predictions_rnn = model_rnn.predict(X_test_reshaped)

# Evaluate RNN model
mse_rnn = mean_squared_error(y_test, predictions_rnn)
print(f'Mean Squared Error (RNN): {mse_rnn}')


# In[36]:


plt.plot(y_test.values, label='Actual')
plt.plot(predictions_rnn, label='Predicted (RNN)')
plt.legend()
plt.title('RNN Model Prediction')
plt.show()


# In[37]:


# Scatter plot of actual vs predicted values
plt.scatter(y_test, predictions_rnn)
plt.title('Actual vs Predicted Scatter Plot (RNN)')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.show()


# In[38]:


residuals_rnn = y_test - predictions_rnn.flatten()

plt.scatter(predictions_rnn.flatten(), residuals_rnn)
plt.title('Residual Plot (RNN)')
plt.xlabel('Predicted Rainfall')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()


# In[39]:


plt.hist(residuals_rnn, bins=30)
plt.title('Distribution of Residuals (RNN)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# # RBFNs Model

# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# In[41]:


from sklearn.kernel_approximation import RBFSampler


# In[42]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[43]:


# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[44]:


# Use RBFSampler for feature transformation
n_components = 100  # You may adjust the number of components
rbf_feature = RBFSampler(gamma=1, n_components=n_components, random_state=42)


# In[45]:


# Transform input features
X_train_rbf = rbf_feature.fit_transform(X_train_scaled)
X_test_rbf = rbf_feature.transform(X_test_scaled)


# In[46]:


# Build RBFN model
rbfn = Ridge(alpha=1.0)
rbfn.fit(X_train_rbf, y_train)


# In[47]:


# Make predictions
predictions_rbfn = rbfn.predict(X_test_rbf)


# In[48]:


# Evaluate the RBFN model
mse_rbfn = mean_squared_error(y_test, predictions_rbfn)
print(f'Mean Squared Error (RBFN): {mse_rbfn}')


# In[49]:


# Define a threshold for correctness
threshold = 0.5  # You may adjust this threshold based on your problem

# Convert predictions to binary values based on the threshold
binary_predictions = (predictions_rbfn >= threshold).astype(int)

# Convert actual values to binary values based on the threshold
binary_actual_values = (y_test >= threshold).astype(int)

# Calculate accuracy
accuracy = (binary_predictions == binary_actual_values).mean()
print(f'Accuracy: {accuracy}')


# In[50]:


import matplotlib.pyplot as plt

# Scatter plot for actual vs predicted values
plt.scatter(y_test, predictions_rbfn, alpha=0.5)
plt.title('Actual vs Predicted Values (RBFN)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# In[51]:


# Residual plot
residuals = y_test - predictions_rbfn
plt.scatter(predictions_rbfn, residuals, alpha=0.5)
plt.title('Residual Plot (RBFN)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # Add a horizontal line at y=0
plt.show()


# In[52]:


# Histogram of residuals
plt.hist(residuals, bins=20, edgecolor='black')
plt.title('Histogram of Residuals (RBFN)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# # MLPs Model

# In[53]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[54]:


# Drop non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])


# In[55]:


# Assuming your data has features X and target variable y
X = numeric_data.drop('ANNUAL', axis=1)  # Adjust column name as needed
y = numeric_data['ANNUAL']


# In[56]:


# Encode categorical variables (if any)
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])


# In[57]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[59]:


# Build the MLP model (similar to the previous example)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))


# In[60]:


# Compile the model
optimizer = Adam(lr=0.001)  # Adjust learning rate as needed
model.compile(optimizer=optimizer, loss='mean_squared_error')


# In[61]:


# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)


# In[62]:


# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on Test Data: {mse}')


# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense


# In[64]:


# Make predictions on the test set
y_pred = model.predict(X_test_scaled)


# In[65]:


# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title('Actual vs. Predicted Rainfall')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.show()


# In[66]:


# Plot a histogram of the residuals
residuals = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.show()


# In[67]:


correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()


# # SOMs Model

# In[68]:


import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[69]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[70]:


# Normalize the data
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)


# In[71]:


# Define the size of the SOM grid
grid_size = (5, 5)  # You can adjust this based on your preference


# In[72]:


# Initialize the SOM
som = MiniSom(grid_size[0], grid_size[1], input_len=X_train.shape[1], sigma=1.0, learning_rate=0.5)


# In[73]:


# Train the SOM
som.train_random(X_train_normalized, 100)  # You can adjust the number of iterations


# In[74]:


# Visualize the SOM
plt.figure(figsize=(8, 8))
for i, x in enumerate(X_train_normalized):
    winner = som.winner(x)
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', markerfacecolor='None', markersize=10, markeredgewidth=2, markeredgecolor='r')
plt.title('Self-Organizing Map')
plt.show()


# In[75]:


# Make predictions on the test set
y_pred = np.array([som.winner(x) for x in X_test_normalized])


# In[76]:


from keras.models import Sequential
from keras.layers import Dense

# Assuming you have already defined and trained your model
model = Sequential()

# Replace 'your_input_dimension' with the actual number of input features in your dataset
input_dimension = 21  

model.add(Dense(units=1, input_dim=21, activation='linear'))
# Add more layers as needed

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (assuming X_train and y_train are your training data)
model.fit(X_train, y_train, epochs=100, batch_size=10)

# Now you can use the trained model to make predictions
y_pred = model.predict(X_test)


# In[77]:


# Assuming X_train is your training data
input_dimension = X_train.shape[1]


# In[78]:


# Evaluate the model (e.g., using mean squared error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# # DBNs Model

# In[79]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


# In[80]:


# Use 'ANNUAL' as the target variable (rainfall to be predicted)
y = data['ANNUAL']


# In[81]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[82]:


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[83]:


# Define and train the DBN using BernoulliRBM
rbm = BernoulliRBM(n_components=256, learning_rate=0.1, n_iter=10, random_state=42, verbose=True)


# In[84]:


# Build the pipeline
dbn_pipeline = Pipeline([('rbm', rbm)])


# In[85]:


# Train the DBN
X_train_dbn = dbn_pipeline.fit_transform(X_train_scaled)


# In[86]:


# Optionally, you can train a regression model on top of the DBN features
from sklearn.linear_model import LinearRegression


# In[87]:


# Check for missing values in X_train_dbn
missing_values = np.isnan(X_train_dbn).any()
print(f'Missing values in X_train_dbn: {missing_values}')


# In[88]:


# If there are missing values, you can handle them by filling with the mean
X_train_dbn = np.nan_to_num(X_train_dbn, nan=np.nanmean(X_train_dbn))


# In[89]:


regression_model = LinearRegression()
regression_model.fit(X_train_dbn, y_train)


# In[90]:


# Make predictions on the test set
X_test_dbn = dbn_pipeline.transform(X_test_scaled)
y_pred = regression_model.predict(X_test_dbn)


# In[91]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[92]:


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


# In[93]:


import seaborn as sns

# Assuming 'df' is your DataFrame
# Replace 'x_column' and 'y_column' with the names of the columns you want to plot
x_column = 'ANNUAL'  # Replace with your actual column name
y_column = 'Minimum temperature (Degree C)'  # Replace with your actual column name

# Scatter plot with Seaborn
sns.scatterplot(x=data[x_column], y=data[y_column], color='blue', alpha=0.5)
plt.title(f'Scatter Plot of {x_column} vs {y_column}')
plt.show()


# # RBMs Model

# In[94]:


import numpy as np
import pandas as pd
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[95]:


print(data.columns)


# In[96]:


data = data.drop(['Quality'], axis=1)


# In[97]:


data = data.drop(['SUBDIVISION'], axis=1)


# In[98]:


print(data.columns)


# In[99]:


# Extract features (X) and target variable (y)
X = data.drop('ANNUAL', axis=1)
y = data['ANNUAL']


# In[100]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[101]:


# Scale the data to the range [0, 1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[102]:


# Define the RBM model
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, batch_size=10, n_iter=10, verbose=1, random_state=42)


# In[103]:


# Create a pipeline with RBM and a linear regression model (or any other model of your choice)
# Here, Linear Regression is used as an example; replace it with your preferred regression model
pipeline = Pipeline(steps=[('rbm', rbm)])


# In[104]:


# Train the pipeline on the training data
pipeline.fit(X_train_scaled)


# In[105]:


# Transform the data using the trained RBM
X_train_rbm = pipeline.transform(X_train_scaled)
X_test_rbm = pipeline.transform(X_test_scaled)


# In[106]:


# Train a linear regression model on the transformed data
from sklearn.linear_model import LinearRegression


# In[107]:


from sklearn.preprocessing import LabelEncoder

# Assuming 'y_train' is categorical, encode it
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Now, you can fit the regression model
regression_model = LinearRegression()
regression_model.fit(X_train_rbm, y_train_encoded)


# In[108]:


# Make predictions on the test set
y_pred = regression_model.predict(X_test_rbm)


# In[109]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[110]:


# Visualize predictions vs. true values
plt.scatter(y_test, y_pred)
plt.xlabel('True Annual Rainfall')
plt.ylabel('Predicted Annual Rainfall')
plt.title('True vs. Predicted Annual Rainfall')
plt.show()


# In[111]:


# Distribution plot for a specific column
plt.figure(figsize=(8, 6))
sns.histplot(data['ANNUAL'], kde=True, color='skyblue')
plt.title('Distribution of Annual Rainfall')
plt.xlabel('Annual Rainfall')
plt.ylabel('Frequency')
plt.show()


# In[112]:


# Assuming 'YEAR' is a column in your DataFrame
plt.figure(figsize=(14, 8))
sns.lineplot(x='ANNUAL', y='ANNUAL', data=data, marker='o')
plt.title('Annual Rainfall Over Time')
plt.xlabel('Annual')
plt.ylabel('Annual Rainfall')
plt.show()


# # AUTO Encoder Model

# In[113]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model


# In[114]:


# Extract features (X) and target variable (y)
X = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].values
y = data['ANNUAL'].values


# In[115]:


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[116]:


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[117]:


# Define the autoencoder model
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(8, activation='relu')(input_layer)
decoded = Dense(X_train.shape[1], activation='linear')(encoded)


# In[118]:


autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')


# In[119]:


# Train the autoencoder
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))


# In[120]:


# Use the trained autoencoder for dimensionality reduction
encoder = Model(input_layer, encoded)
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)


# In[121]:


# Display the architecture of the autoencoder
autoencoder.summary()


# In[122]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[123]:


# Define a regression model (you can use any other regression model as well)
regression_model = LinearRegression()


# In[124]:


# Fit the regression model on the encoded training data
regression_model.fit(X_train_encoded, y_train)


# In[125]:


# Make predictions on the encoded test data
y_pred = regression_model.predict(X_test_encoded)


# In[126]:


# Evaluate the performance of the regression model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[127]:


# Assuming you have y_new_actual, replace it with your actual labels
# y_new_actual = ...
import matplotlib.pyplot as plt
# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter([10.5, 15.2, 20.0], [13.5, 17.2, 21.0], color='blue', label='Actual vs Predicted')
plt.plot([min(10.5, 15.2, 20.0), max(13.5, 17.2, 21.0)], [min(10.5, 15.2, 20.0), max(13.5, 17.2, 21.0)], linestyle='--', color='red', label='Ideal Line')
plt.title('Actual vs Predicted Values on New Encoded Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()


# In[128]:


# Visualize the original vs. reconstructed data
encoded_data = autoencoder.predict(X_train)


# In[129]:


n = 10  # Number of samples to visualize
plt.figure(figsize=(20, 6))
for i in range(n):
    # Original data
    plt.subplot(2, n, i + 1)
    plt.imshow(X_train[i].reshape(1, -1), cmap='gray')
    plt.title("Original")
    plt.axis('off')
 # Reconstructed data
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_data[i].reshape(1, -1), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()


# In[ ]:




