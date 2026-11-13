
# CUSTOMER CHURN ANALYSIS PROJECT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create sample dataset
np.random.seed(42)

n = 200

data = {
    'CustomerID': range(1, n + 1),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 65, n),
    'Tenure': np.random.randint(1, 73, n),
    'MonthlyCharges': np.random.randint(20, 120, n),
    'Contract': np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], n
    ),
    'InternetService': np.random.choice(
        ['DSL', 'Fiber optic', 'No'], n
    )
}

df = pd.DataFrame(data)

# Step 2: Generate realistic churn values
churn_probability = np.where(
    df['Contract'] == 'Month-to-month', 0.45, 0.15
)

churn_probability += np.where(df['Tenure'] < 12, 0.15, 0)
churn_probability = np.clip(churn_probability, 0, 0.9)

df['Churn'] = np.where(
    np.random.random(n) < churn_probability,
    'Yes', 'No'
)

# Step 3: Save original dataset
df.to_csv('customer_data.csv', index=False)

print("ORIGINAL CUSTOMER DATA")
print(df.head())

# Step 4: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Remove duplicate records
df = df.drop_duplicates()

# Step 6: Handle missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Step 7: Correct inconsistent formats
df['Gender'] = df['Gender'].str.strip().str.title()
df['Contract'] = df['Contract'].str.strip()
df['InternetService'] = df['InternetService'].str.strip()

# Step 8: Calculate churn rate
churn_count = df['Churn'].value_counts()

churn_rate = (
    (df['Churn'] == 'Yes').sum() / len(df)
) * 100

print("\nCUSTOMER CHURN COUNT")
print(churn_count)

print("\nCustomer Churn Rate:")
print(round(churn_rate, 2), "%")

# Step 9: Average monthly charges
print("\nAverage Monthly Charges:")
print(df['MonthlyCharges'].mean())

# Step 10: Churn by contract
print("\nChurn by Contract:")
print(pd.crosstab(df['Contract'], df['Churn']))

# Step 11: Churn by gender
print("\nChurn by Gender:")
print(pd.crosstab(df['Gender'], df['Churn']))

# Step 12: Create visualizations
sns.set_theme(style="whitegrid")

# Graph 1: Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Churn')
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.savefig('churn_distribution.png')
plt.show()

# Graph 2: Churn by contract
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Customer Churn by Contract')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.savefig('churn_by_contract.png')
plt.show()

# Graph 3: Monthly charges
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges vs Customer Churn')
plt.savefig('monthly_charges.png')
plt.show()

# Graph 4: Tenure analysis
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df, x='Tenure', hue='Churn',
    bins=15, multiple='stack'
)
plt.title('Customer Tenure Analysis')
plt.xlabel('Tenure (Months)')
plt.savefig('tenure_analysis.png')
plt.show()

# Step 13: Structural validation
print("\nSTRUCTURAL VALIDATION")
print("Dataset Shape:", df.shape)
print("Duplicate Rows:", df.duplicated().sum())
print("Missing Values:", df.isnull().sum().sum())
print("Valid Churn Values:", df['Churn'].isin(['Yes', 'No']).all())

# Step 14: Save final cleaned dataset
df.to_csv('cleaned_customer_data.csv', index=False)

# Step 15: Final result
print("\nFINAL RESULT")
print("Total Customers:", len(df))
print("Churned Customers:", (df['Churn'] == 'Yes').sum())
print("Retained Customers:", (df['Churn'] == 'No').sum())
print("Churn Rate:", round(churn_rate, 2), "%")

print("\nCustomer Churn Analysis Completed Successfully!")
