'''
This Python code establishes Linear Regression (unsuccessful) and KNN Regression (successful) models based on the Modelled NOx
concentration dataset found in the LAEI repository.

There are three different methods to identify outliers with comments on when and how to utilize them - We will most likely
need one or more of them for pre-processing the data.
'''

# Importing modules and packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 
from math import sqrt
  
# Importing data 
df = pd.read_csv('/4.1. Concentrations LAEI 2013 Update/PostLAEI2013_2013_NOx.csv') 
df.drop('year', inplace=True, axis=1) 
  
print(df.head()) 
print(df.shape)

# Check skewness
print(df['conct'].skew())

# Gain insight into data and outliers (Compare mean vs. max)
df.describe()[['conct']]
sns.boxplot(df['conct'])

# Option 1 - Remove outliers (z-score < 3) --- ONLY WHEN DATA FOLLOWS NORMAL DISTRIBUTION!!!

lim = np.abs((df - df.mean()) / df.std(ddof=0)) < 3
df.loc[:, df.columns] = df.where(lim, np.nan)
df.dropna(inplace=True)

# Option 2: Percentile Method (Always used symmetrically, eg 0.01 - 0.99, 0.02 - 0.98, etc.)
lim = np.logical_and(df < df.quantile(0.99, numeric_only=False),
                     df > df.quantile(0.01, numeric_only=False))
df.loc[:, df.columns] = df.where(lim, np.nan)
df.dropna(inplace=True)

# Option 3: IQR - most popular method
Q1 = np.percentile(df['conct'], 25, method='midpoint')
Q3 = np.percentile(df['conct'], 75, method='midpoint')

IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

upper_array = np.where(df['conct'] >= upper)[0]
lower_array = np.where(df['conct'] <= lower)[0]

df.drop(index=upper_array, inplace=True)
df.drop(index=lower_array, inplace=True)


# Plotting Histogram (kde = kernel density)
sns.histplot(df['conct'], kde=True).set(title='NOx Concentration Histogram')

# plotting scatterplot 
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = df['x']
y = df['y']
z = df['conct']

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("concentration")

ax.scatter(x, y, z)
plt.figure(figsize=(15,15))

plt.show()
  
# creating feature variables 
X = df.drop('conct', axis=1) 
y = df['conct'] 
  
print(X) 
print(y) 
  
# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=50) 
  
# creating a linear regression model 
model = LinearRegression() 
  
# fitting the model 
model.fit(X_train, y_train) 
  
# making predictions 
predictions = model.predict(X_test) 

# Same for KNN regression
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Find optimum number of k
rmse_val = []
for k in range(1, 10):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    error = sqrt(mean_squared_error(y_test, predictions)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print(f'Score for k={k}: {model.score(X_test, y_test)}')

# Print elbow
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()

# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 
model.score(X_test, y_test)
