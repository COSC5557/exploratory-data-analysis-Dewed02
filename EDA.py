import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
pima = pd.read_csv(r'C:\Users\Derek\OneDrive\Desktop\archive\diabetes.csv', sep=',')

# Overview of data
print(pima.head())
print(pima.info()) # 8 features, 1 target, no null values
print(pima.shape) # 768 observations, 9 features
print(pima.describe()) # Possible outliers in Insulin, SkinThickness, and BMI
print(pima.Outcome.value_counts()) # 500 non-diabetic, 268 diabetic

# Check for correlation
plt.figure(figsize=(20,15))
sns.heatmap(pima.corr(), annot=True, cmap='RdYlGn')
plt.show()

# Check for outliers   
plt.figure(figsize=(20,15))
pima.boxplot(column=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], rot=45)
plt.show()

# Check for skewness
plt.figure(figsize=(20,15))
pima.hist()
plt.show()

# Check for missing values
plt.figure(figsize=(20,15))
sns.heatmap(pima.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()