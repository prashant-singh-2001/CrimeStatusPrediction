import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler

df = pd.read_csv("Cleaned_Data.csv")

df.isnull().any()
df['Date Rptd'] = pd.to_datetime(df['Date Rptd']).values.astype(float) //10**9
df['DATE OCC'] = pd.to_datetime(df['DATE OCC']).values.astype(float) //10**9
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)
scaler=StandardScaler()
new_data=pd.DataFrame(scaler.fit_transform(df))
label_enc=LabelEncoder()
df.iloc[:,8]=label_enc.fit_transform(df.iloc[:,8])
df.iloc[:,-1]=label_enc.fit_transform(df.iloc[:,-1])


plt.figure(figsize=(15, 10))

plt.pie(df['Crm Cd'].value_counts(), autopct='%.2f',labels=df['Crm Cd'].unique())

plt.title('Crime Count Pie Chart')

# Set the figure size of the plot to 10x5 inches
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True);


