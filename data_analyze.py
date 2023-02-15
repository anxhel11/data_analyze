#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, delim_whitespace=True)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Explore the dataset
sns.pairplot(df, x_vars=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX'], y_vars=['MEDV'])
sns.pairplot(df, x_vars=['RM', 'AGE', 'DIS', 'RAD', 'TAX'], y_vars=['MEDV'])
sns.pairplot(df, x_vars=['PTRATIO', 'B', 'LSTAT'], y_vars=['MEDV'])
plt.show()

# Preprocess the data
df.dropna(inplace=True)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluate the model's performance
print('Coefficients:', regressor.coef_)
print('Mean squared error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

# Visualize the model's predictions
plt.scatter(X_test[:, 3], y_test, color='red')
plt.plot(X_test[:, 3], y_pred, color='blue')
plt.show()

# Use the model to predict the median value of owner-occupied homes for new neighborhoods
new_neighborhood = [[0, 1, 0, 0, 6.0, 50, 6.0, 5.0, 1, 200, 15, 500, 5]]
new_neighborhood = np.array(ct.transform(new_neighborhood))
new_neighborhood = sc.transform(new_neighborhood)
new_pred = regressor.predict(new_neighborhood)
print('Predicted median value of owner-occupied homes for the new neighborhood:', new_pred)


# In[1]:


#This is a data analysis project that uses Python to analyze the Boston Housing dataset. The goal of the project is to build a linear regression model that can predict the median value of owner-occupied homes based on various features, such as crime rate, number of rooms, and student-teacher ratio.

#We start by loading the dataset into a Pandas DataFrame and exploring the data using visualizations. Then, we preprocess the data by handling missing values, one-hot encoding categorical features, and standardizing the data using a scaler. Next, we split the data into training and testing sets and build a linear regression model using scikit-learn's LinearRegression class.

#We evaluate the model's performance using metrics like mean squared error and R-squared, and visualize its predictions using a scatter plot. Finally, we use the model to predict the median value of owner-occupied homes for a new neighborhood based on its features.

#Overall, this project demonstrates how Python can be used to perform data analysis and build machine learning models for real-world problems. It also showcases several important data analysis and machine learning techniques, such as data preprocessing, feature engineering, model selection, and performance evaluation.

