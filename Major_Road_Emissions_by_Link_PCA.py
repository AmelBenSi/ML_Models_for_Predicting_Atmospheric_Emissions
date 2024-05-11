import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_excel('3 -LAEI2013_MajorRoads_EmissionsbyLink_2013.xlsx') 

df.columns

df_proc = df.replace(0,np.nan)
df_proc.isnull().sum()
df_proc.dtypes

df_proc.drop(['ElectricLgv', 
              'ElectricCar', 
              'Coach', 
              'LtBus', 
              'BusAndCoach', 
              'Motorcycle', 
              'GridId', 
              'Toid', 
              'GRID_ExactCut_ID', 
              'Location_ExactCut', 
              'BoroughName_ExactCut', 
              'Emissions Unit', 
              'Year', 
              'Emissions', 
              'Lts'], 
             axis=1, inplace=True)

df_proc = df_proc.dropna()
df_proc.describe

df_proc.head(1)


# Encoding Pollutants
pollutant_col = df_proc[["Pollutant"]]
encoder = OrdinalEncoder().set_output(transform="pandas")
pollutant_encoded = encoder.fit_transform(pollutant_col)

encoder.categories_

df_encoded = encoder.fit_transform(df_proc)
df_encoded.dtypes

# Scale dataset
std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(df_encoded)

# Create PCA
pca = PCA(n_components=3)
x = pca.fit_transform(scaled_df)
print(sum(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_)

# Find most relevant features for PC1, PC2, and PC3
print(abs(pca.components_)) 

# Generate loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', "PC3"], index=df_encoded.keys())
loadings

# Generate plots
nums = np.arange(8)
var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(scaled_df)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))
  
# Plot 1
plt.figure(figsize=(8,4))
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Components vs. Explained Variance Ratio')

# Plot 2
plt.figure(figsize=(8,4))
plt.grid()
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Component vs. Explained Variance Ratio')

# Plot 3
plt.figure(figsize=(8,4))
plt.grid()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio');
plt.title('Number of Components vs. Explained Variance Ratio')

# 2D PCA Plot
plt.figure(figsize=(20, 15))
 
plot = plt.scatter(x[:, 0], x[:, 1],
            c=df_encoded['Pollutant'],
            cmap='plasma')
 
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(handles=plot.legend_elements()[0], labels=list(df_encoded['Pollutant']))
plt.show()

# 3D PCA Plot
fig = plt.figure(figsize = (20, 15))
ax = plt.axes(projection ="3d")
 
ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], 
             c = df_encoded['Pollutant'],
             cmap = 'plasma')
plt.show()
