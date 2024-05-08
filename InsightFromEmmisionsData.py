#!/usr/bin/env python
# coding: utf-8

# In[2]:


##import Concentration data from dataframes
import pandas as pd
import os

def load_csvs_to_dfs(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    data_frames = {}
    for file in csv_files:
        df_name = file[:-4]  # Removing the '.csv' part to use as the dictionary key
        path = os.path.join(directory, file)
        data_frames[df_name] = pd.read_csv(path)
    return data_frames

#point to the CSV files that are located in a folder named 'data_folder' in the current directory
directory_path = 'C://Users//tayla//Documents//Emmisions_Data//'
data_frames_dict = load_csvs_to_dfs(directory_path)


# In[5]:


# Updating DataFrames and storing them separately
all_data_frames = []  # List to hold all data frames for later concatenation
for df_name, df in data_frames_dict.items():
    # Extract the pollutant type from the DataFrame name
    pollutant = df_name.split('_')[2]
    df['Pollutant'] = pollutant
    all_data_frames.append(df.copy())  # Append a copy of the updated DataFrame to the list
    data_frames_dict[df_name] = df  # Update the dictionary with the modified DataFrame
    print(f"Updated DataFrame {df_name}:")
    print(df.head())  # Print the first few rows of each updated DataFrame

# Concatenate all updated DataFrames into one
combined_df_vertical = pd.concat(all_data_frames, ignore_index=True)

# Print the head of the combined DataFrame to verify
print("Combined DataFrame Preview:")
print(combined_df_vertical.head())


# In[7]:


available_dfs = list(data_frames_dict.keys())
print("Available DataFrames:", available_dfs)


# In[8]:


#example knn ml model on No2 data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# demonstrate on dataframe
df = data_frames_dict['PostLAEI2013_2013_NO2']  # Use the appropriate key for your DataFrame

# Prepare the feature matrix and target vector
X = df[['x', 'y']]  # Features are the coordinates
y = df['conct']     # Target is the concentration

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the KNN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5) 
knn_regressor.fit(X_train, y_train)

# Making predictions
y_pred = knn_regressor.predict(X_test)

# Evaluating the model (u
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[ ]:


##plotting 3d plots for data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to plot each DataFrame
def plot_3d_scatter(df, title):
    fig = plt.figure(figsize=(10, 8))  # Set the figure size
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Scatter plot using x, y, and conct columns
    scatter = ax.scatter(df['x'], df['y'], df['conct'], c=df['conct'], cmap='viridis', alpha=0.6)

    # Labeling
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Concentration')
    ax.set_title(title)

    # Color bar to show concentration levels
    cbar = fig.colorbar(scatter, pad=0.1)
    cbar.set_label('Concentration')

    # Show plot
    plt.show()

# Iterate over each DataFrame and create a plot
for name, df in data_frames_dict.items():
    plot_3d_scatter(df, f"3D Scatter Plot of {name}")


# In[ ]:




