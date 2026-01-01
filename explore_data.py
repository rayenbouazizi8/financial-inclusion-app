import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("financial_inclusion.csv")

# Basic info
print("\n🔍 Data Info:")
print(df.info())

print("\n📊 First 5 Rows:")
print(df.head())

print("\n📈 Summary Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

print("\n🧮 Unique Values per Column:")
print(df.nunique())

print("\n✅ Target Value Counts (HasBankAccount):")
print(df["HasBankAccount"].value_counts())

# Optional visual
sns.countplot(data=df, x="HasBankAccount")
plt.title("Target Distribution")
plt.show()
