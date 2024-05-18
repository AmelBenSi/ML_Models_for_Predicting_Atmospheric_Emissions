# Importing Libraries and Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Reading Raw Data into dataframe
df = pd.read_excel('3 -LAEI2013_MajorRoads_EmissionsbyLink_2013.xlsx') 

# Creating a copy for further processing
df_proc = df.copy()
df_proc.columns

# Summing up emissions by vehicle types
column_names = list(df_proc.columns[11:])
df_proc['Link_Emissions'] = df_proc[column_names[:]].sum(axis=1)

df_proc.drop(column_names, axis=1, inplace=True)

# Dropping unused columns
df_proc.drop(['Length (m)', 
              'Location_ExactCut',
              'GRID_ExactCut_ID',
              'Emissions Unit', 
              'GridId',
              'Year', 
              'Lts',
              'Emissions',
              ], axis=1, inplace=True)


# Create Pivot table to have Emission types as columns
df_proc = df_proc.pivot_table(index=['Toid', 'BoroughName_ExactCut'], columns=['Pollutant'], values=['Link_Emissions'])
df_proc.columns = df_proc.columns.droplevel(0)
df_proc = df_proc.reset_index()

# Converting 0 values to NaN and dropping rows containing NaN vals
df_proc = df_proc.replace(0,np.nan)
df_proc = df_proc.dropna()

# Checking for outliers
sns.boxplot(data=df_proc[df_proc.columns[2:]])

# Removing the outliers
def dropOutliers(data, col):
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)
    IQR = Q3 - Q1
 
    print("IQR value for column %s is: %s" % (col, IQR))
    global outlier_free_list
    global filtered_data
 
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    outlier_free_list = [x for x in data[col] if (
        (x > lower_range) & (x < upper_range))]
    filtered_data = data.loc[data[col].isin(outlier_free_list)]
 
 
for i in df_proc.columns[2:]:
      if i == df_proc.columns[0]:
          dropOutliers(df_proc, i)
      else:
          dropOutliers(filtered_data, i)

df_proc = filtered_data

# Extract Borough column to separate variable and delete Borough and Toid columns
y = df_proc['BoroughName_ExactCut']
df_proc.drop(['BoroughName_ExactCut', 'Toid'], axis=1, inplace=True)

df_proc.columns

df_proc.describe()

# Scaling the data
std_scaler = StandardScaler()
df_scaled = std_scaler.fit_transform(df_proc)

# Generate Scree Plot
pca = PCA()
x = pca.fit_transform(df_scaled)
explained_variance = pca.explained_variance_ratio_
plt.plot(pca.explained_variance_, marker='o')
plt.xlabel("Eigenvalue number")
plt.ylabel("Eigenvalue size")
plt.title("Scree Plot")

# Generate models
nums = np.arange(8)
var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(df_scaled)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))
  
# Plot PCA Components vs. Explained Variance Ratio
plt.figure(figsize=(8,4))
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Components vs. Explained Variance Ratio')

# Create two component PCA
pca = PCA(n_components=2)
x = pca.fit_transform(df_scaled)
print(sum(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_)

# Generate loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=df_proc.columns)
print(f'Loadings:\n{loadings_df}\n===============================')

# Create DataFrame with principal components and target (Borough)
PCA_df = pd.DataFrame(data = x)
PCA_df = PCA_df.reset_index(drop=True)
y = y.reset_index(drop=True)
df_final = pd.concat([PCA_df, y], axis = 1)
df_final.columns = ['PC1', 'PC2', 'Borough']

# 2D PCA Plot with hue by Borough
sns.set(font_scale=2.0)
sns.set_theme(rc={'figure.figsize':(30,15)})
sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='Borough')
plt.title("PC1 vs. PC2")
