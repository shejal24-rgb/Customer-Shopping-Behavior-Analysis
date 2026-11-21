
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\ProjectDA\Super-Market-Sales-Analysis-.py\customer_shopping.csv")

df = df.drop_duplicates()

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["Price"] = df["Price"].fillna(df["Price"].median())

df["Revenue"] = df["Quantity"] * df["Price"]

df["Age_Group"] = np.where(
    df["Age"] < 30, "Young",
    np.where(df["Age"] < 45, "Adult", "Senior")
)

print("Dataset Overview")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nTotal Revenue:", df["Revenue"].sum())
print("Average Purchase Value:", round(df["Revenue"].mean(), 2))
print("Total Customers:", df["Customer_ID"].nunique())
print("Total Products Purchased:", df["Quantity"].sum())

category_sales = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
print("\nCategory-wise Revenue")
print(category_sales)

gender_sales = df.groupby("Gender")["Revenue"].sum()
print("\nGender-wise Revenue")
print(gender_sales)

payment_sales = df.groupby("Payment_Method")["Revenue"].sum().sort_values(ascending=False)
print("\nPayment Method Analysis")
print(payment_sales)

monthly_sales = df.groupby("Shopping_Month")["Revenue"].sum()
print("\nMonthly Revenue")
print(monthly_sales)

age_analysis = df.groupby("Age_Group")["Revenue"].sum()
print("\nAge Group Analysis")
print(age_analysis)

product_sales = df.groupby("Product")["Quantity"].sum().sort_values(ascending=False)
print("\nPopular Products")
print(product_sales)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title("Category-wise Revenue")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title("Monthly Shopping Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=gender_sales.index, y=gender_sales.values)
plt.title("Gender-wise Shopping Revenue")
plt.xlabel("Gender")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(x=payment_sales.index, y=payment_sales.values)
plt.title("Revenue by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=age_analysis.index, y=age_analysis.values)
plt.title("Age Group-wise Revenue")
plt.xlabel("Age Group")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Popular Products by Quantity")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nBusiness Insights")
print("Top Revenue Category:", category_sales.idxmax())
print("Highest Revenue Gender:", gender_sales.idxmax())
print("Most Used Payment Method by Revenue:", payment_sales.idxmax())
print("Best Revenue Month:", monthly_sales.idxmax())
print("Most Popular Product:", product_sales.idxmax())
print("Highest Revenue Age Group:", age_analysis.idxmax())
