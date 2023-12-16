import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
# data base link : https://catalog.data.gov/dataset/crime-data-from-2020-to-present

# Reading the dataset
data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
print(data.shape)

# Dropping specified columns
data = data.drop(['DR_NO','Crm Cd Desc','AREA NAME','Premis Desc','Weapon Desc','Status Desc','Mocodes','LAT','LON','LOCATION','Cross Street','Vict Descent','Crm Cd 1','Crm Cd 2','Crm Cd 3','Crm Cd 4'], axis=1)

# Calculating value counts for crime codes
crime_counts = data['Crm Cd'].value_counts()

# Define a threshold for the minimum number of occurrences
threshold = 1000

# Create a list of crime codes to remove based on the threshold
negligible_crime_codes = crime_counts[crime_counts <= threshold].index.tolist()

# Remove the rows where the crime code is in the negligible list
data = data[~data['Crm Cd'].isin(negligible_crime_codes)]

print(data.shape)

# Converting date columns to datetime and handling null values
data['Date Rptd'] = pd.to_datetime(data['Date Rptd']).values.astype(float) // 10**9
data['DATE OCC'] = pd.to_datetime(data['DATE OCC']).values.astype(float) // 10**9
data.isnull().any()
data.fillna(method="ffill", inplace=True)  # Forward fill null values
data.fillna(method="bfill", inplace=True)  # Backward fill null values

# Encoding categorical variables using LabelEncoder
label_enc = LabelEncoder()
data['Status'] = label_enc.fit_transform(data['Status'])
data = data[~data['Status'].isin([2, 4, 5])]  # Removing specific status values
data.iloc[:, 8] = label_enc.fit_transform(data.iloc[:, 8])  # Encoding a specific column

# Columns to be scaled
columns_to_scale = ['Date Rptd', 'DATE OCC', 'TIME OCC', 'AREA', 'Rpt Dist No', 'Vict Age']

# Selecting columns for scaling
data_to_scale = data[columns_to_scale]

# Creating a copy of the original DataFrame to retain non-standardized columns
data_remaining = data.drop(columns_to_scale, axis=1)

# Standardizing selected columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)

# Creating a DataFrame with the scaled columns
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

# Resetting indices for concatenation
scaled_df.reset_index(drop=True, inplace=True)
data_remaining.reset_index(drop=True, inplace=True)

# Combining the scaled columns with the remaining columns
final_data = pd.concat([scaled_df, data_remaining], axis=1)

# Saving the cleaned data to a new CSV file
final_data.to_csv('Cleaned_Data.csv', index=False)
