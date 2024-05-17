# ML_Models_for_Predicting_Atmospheric_Emissions

This repository contains the code and data for our end-of-module assignment as part of an MSc in Data Science and AI. The project is implemented in Python using Jupyter Notebook.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Project Structure](#Project-Structure)
- [Data Files](#Data-files)
- [Usage](#usage)
- [Installation](#installation)
- [License](#license)

## Project Description

This project aims to develop and compare various machine learning models to predict atmospheric emissions. The models are built using Scikit-learn, utilizing data from various London boroughs and county environment monitoring units. The project consists of data preprocessing, exploratory data analysis, model training, and evaluation using regression techniques and Principal Component Analysis (PCA).

## Features

- **Data Preprocessing:** Standardization and outlier removal
- **Exploratory Data Analysis:** Visualization of data distributions and correlations
- **Regression Models:** Implementation of Decision Tree, K-Nearest Neighbors, Support Vector Regression (SVR), and Bagging Regressors
- **Dimensionality Reduction:** PCA for feature extraction in high-dimensional data

## Project Structure

```bash
ML_Models_for_Predicting_Atmospheric_Emissions/
├── data/
│   ├── csv/
│   │   ├── 4_PostLAEI2013_2013_NO2.csv
│   │   ├── 4_PostLAEI2013_2013_NOx.csv
│   │   ├── 4_PostLAEI2013_2013_PM10.csv
│   │   ├── 4_PostLAEI2013_2013_PM10d.csv
│   │   └── 4_PostLAEI2013_2013_PM25.csv
│   └── excel/
│       └── LAEI2013_MajorRoads_EmissionsbyLink_2013.xlsx
└── src/
    ├── reg.ipynb
    └── pca.ipynb
```

## Data Files

Ensure the following data files are placed in the appropriate directories:

### CSV Folder (`data/csv`):

- `4_PostLAEI2013_2013_NO2.csv`
- `4_PostLAEI2013_2013_NOx.csv`
- `4_PostLAEI2013_2013_PM10.csv`
- `4_PostLAEI2013_2013_PM10d.csv`
- `4_PostLAEI2013_2013_PM25.csv`

### Excel Folder (`data/excel`):

- `LAEI2013_MajorRoads_EmissionsbyLink_2013.xlsx`

### Data Source

The data was sourced from the [London Atmospheric Emissions Inventory 2013] (https://data.london.gov.uk/dataset/london-atmospheric-emissions-inventory-2013).


## Usage

To use this project, follow the step below to clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/AmelBenSi/ML_Models_for_Predicting_Atmospheric_Emissions.git
   cd ML_Models_for_Predicting_Atmospheric_Emissions
```

## Installation

To run this project, ensure you have the following packages installed:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## License 
This project is licensed under the MIT License.
