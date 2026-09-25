import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0. global styles setup

# Folder where all charts will be saved
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

def save_chart(filename):
    """Save the current chart and close it."""
    path = f"{CHART_DIR}/{filename}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f" saved -> {path}")
    
# 1. Load data

title = "1. DATASET DESCRIPTION"
print(f"\n{title:=^80}\n")

# Load the retail customer dataset
df = pd.read_csv("KBIDA421_Assignment_1_Retail_Customer_Dataset.csv")

print(f"Observations: {df.shape[0]} | Variables: {df.shape[1]}")
df.info()

# Define the main variable groups
numeric_cols = [
    "Age",
    "Monthly_Income",
    "Monthly_Spending",
    "Number_of_Purchases",
    "Website_Visits",
    "Support_Calls",
]

categorical_cols = [
    "Payment_Method",
    "Region",
    "Previous_Campaign_Response",
]

target_col = "Churn"
id_col = "Customer_ID"

# 2. TARGET VARIABLE

title = "2. TARGET VARIABLE"
print(f"\n{title:=^80}\n")

print("""
Churn is the target variable with binary yes/no
""")

# 3. DATA-QUALITY ASSESSMENT

title = "3. DATA-QUALITY ASSESSMENT"
print(f"\n{title:=^80}\n")

# 3a. Check for missing values
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct,
})

missing_report = missing_report[missing_report["missing_count"] > 0]

print("Missing values by column:")
print(missing_report)

# 3b. Check for duplicate rows and duplicate Customer_IDs
print(f"\nFully duplicated rows: {df.duplicated().sum()}")
print(f"Duplicated Customer_ID rows: {df.duplicated(subset=[id_col]).sum()}")

# Show duplicate customer records together
dup_cols = [id_col, "Age", "Monthly_Income", "Churn"]

dup_view = (
    df[df.duplicated(subset=[id_col], keep=False)]
    .sort_values(id_col)[dup_cols]
)

print("\nDuplicate ID records (Grouped):")
print(dup_view.set_index([id_col, "Age"]))

# 3c. Check for invalid values
invalid_age = ((df["Age"] < 10) | (df["Age"] > 100)).sum()
invalid_income = (df["Monthly_Income"] <= 0).sum()
invalid_spending = (df["Monthly_Spending"] <= 0).sum()

print(f"\nInvalid Age (<10 or >100): {invalid_age}")
print(f"Invalid Monthly_Income (<=0): {invalid_income}")
print(f"Invalid Monthly_Spending (<=0): {invalid_spending}")

# 3d. Check unusual / extreme observations
print("\nTop 5 highest Monthly_Income values:")
print(df["Monthly_Income"].sort_values(ascending=False).head(5).tolist())

print("Top 5 highest Monthly_Spending values:")
print(df["Monthly_Spending"].sort_values(ascending=False).head(5).tolist())

# Remove duplicate Customer_ID records before continuing
df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)

print(
    f"Rows after removing duplicate Customer_ID records: {len(df)}"
)

# Calculate churned data on cleaned data: the 500 not 503
churn_counts = df[target_col].value_counts()
churn_rate = (df[target_col] == "Yes").mean() * 100
print(f"\nChurn counts on cleaned {len(df)}-customer dataset:")
print(churn_counts)
print(f"Churn rate: {churn_rate:.1f}%")

# ============================================================
# 4. DESCRIPTIVE STATISTICS

title = "4. DESCRIPTIVE STATISTICS"
print(f"\n{title:=^80}\n")

# Summary statistics plus skewness for numeric variables
desc = df[numeric_cols].describe()
desc["skew"] = df[numeric_cols].skew()
desc = desc.round(2)

print(desc)

# ============================================================
# 5. EXPLORATORY VISUALISATIONS

title = "5. EXPLORATORY VISUALISATIONS"
print(f"\n{title:=^80}\n")

# Visualisation 1: Histogram of Age
plt.figure(figsize=(6, 4))
plt.hist(df["Age"].dropna(), bins=15, edgecolor="black")
plt.title("Distribution of Customer Age")
plt.xlabel("Age (years)")
plt.ylabel("Number of customers")
save_chart("01_age_histogram")


# Visualisation 2: Income and spending boxplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].boxplot(df["Monthly_Income"].dropna())
axes[0].set_title("Monthly Income")
axes[0].set_ylabel("USD")

axes[1].boxplot(df["Monthly_Spending"].dropna())
axes[1].set_title("Monthly Spending")
axes[1].set_ylabel("USD")

fig.suptitle("Boxplots: Monthly Income and Monthly Spending")
save_chart("02_income_spending_boxplots")


# Visualisation 3: Churn rate by region
region_churn = (
    df.groupby("Region")[target_col]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(7, 4.5))
plt.bar(region_churn.index, region_churn.values, edgecolor="black")
plt.title("Churn Rate by Region")
plt.xlabel("Region")
plt.ylabel("Churn rate (%)")
plt.xticks(rotation=45, ha="right")
save_chart("03_churn_rate_by_region")


# Visualisation 4: Churn rate by previous campaign response
camp_churn = (
    df.groupby("Previous_Campaign_Response")[target_col]
    .apply(lambda s: (s == "Yes").mean() * 100)
)

plt.figure(figsize=(5, 4))
plt.bar(camp_churn.index, camp_churn.values, edgecolor="black")
plt.title("Churn Rate by Previous Campaign Response")
plt.xlabel("Previous Campaign Response")
plt.ylabel("Churn rate (%)")
save_chart("04_churn_rate_by_campaign_response")

print(
    "4 visualisations produced "
    "(Age histogram, income spending boxplot, churn rate by region "
    "and churn rate by campaign response)."
)

# 6. RELATIONSHIPS BETWEEN VARIABLES

title = "6. RELATIONSHIPS BETWEEN VARIABLES"
print(f"\n{title:=^80}\n")

# Convert Churn to numbers so correlations can be calculated:
# No = 0, Yes = 1
df_churn_num = df.assign(
    Churn_num=(df["Churn"] == "Yes").astype(int)
)

corr_income_spending = df["Monthly_Income"].corr(df["Monthly_Spending"])

corr_calls_churn = df_churn_num["Support_Calls"].corr(df_churn_num["Churn_num"])

corr_purchases_churn = df_churn_num["Number_of_Purchases"].corr(df_churn_num["Churn_num"])

corr_visits_churn = df_churn_num["Website_Visits"].corr(df_churn_num["Churn_num"])

print(
    f"Correlation - Monthly_Income vs Monthly_Spending: "
    f"{corr_income_spending:.3f}"
)

print(
    f"Correlation - Support_Calls vs Churn (numeric): "
    f"{corr_calls_churn:.3f}"
)

print(
    f"Correlation - Number_of_Purchases vs Churn (numeric): "
    f"{corr_purchases_churn:.3f}"
)

print(
    f"Correlation - Website_Visits vs Churn (numeric): "
    f"{corr_visits_churn:.3f}"
)

# 7. MISSING VALUE HANDLING

title = "7. MISSING VALUE HANDLING"
print(f"\n{title:=^80}\n")

# Income and spending are right-skewed, so use the median
income_median = df["Monthly_Income"].median()
spending_median = df["Monthly_Spending"].median()

df["Monthly_Income"] = df["Monthly_Income"].fillna(income_median)
df["Monthly_Spending"] = df["Monthly_Spending"].fillna(spending_median)

# Payment_Method is categorical, so use the most common value (mode)
payment_mode = df["Payment_Method"].mode()[0]
df["Payment_Method"] = df["Payment_Method"].fillna(payment_mode)

print(f"Monthly_Income missing values filled with median: {income_median:.2f}")
print(f"Monthly_Spending missing values filled with median: {spending_median:.2f}")
print(f"Payment_Method missing values filled with mode: '{payment_mode}'")
print("Missing values remaining:", df.isna().sum().sum())

# 8. OUTLIER DETECTION (IQR RULE)

title = "8. OUTLIER DETECTION (IQR RULE)"
print(f"\n{title:=^80}\n")

def iqr_bounds(series):
    """Return the lower and upper IQR outlier limits."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

# Find outliers for income and spending
for col in ["Monthly_Income", "Monthly_Spending"]:
    low, high = iqr_bounds(df[col])
    outliers = ((df[col] < low) | (df[col] > high)).sum()

    print(
        f"{col}: IQR bounds = [{low:.2f}, {high:.2f}], "
        f"outliers = {outliers}"
    )

# Display the values above the upper IQR limit
low_inc, high_inc = iqr_bounds(df["Monthly_Income"])
print(
    "\nOutlier Monthly_Income values:",
    df.loc[df["Monthly_Income"] > high_inc, "Monthly_Income"].tolist(),
)

low_sp, high_sp = iqr_bounds(df["Monthly_Spending"])
print(
    "Outlier Monthly_Spending values:",
    df.loc[df["Monthly_Spending"] > high_sp, "Monthly_Spending"].tolist(),
)

# Cap extreme values instead of deleting the customer records (aka winsoring)
df["Monthly_Income"] = np.where(
    df["Monthly_Income"] > high_inc,
    high_inc,
    df["Monthly_Income"],
)

df["Monthly_Spending"] = np.where(
    df["Monthly_Spending"] > high_sp,
    high_sp,
    df["Monthly_Spending"],
)

# 9. CATEGORICAL ENCODING

title = "9. CATEGORICAL ENCODING"
print(f"\n{title:=^80}\n")

# Binary encoding: Yes = 1, No = 0
df["Previous_Campaign_Response_enc"] = (
    df["Previous_Campaign_Response"] == "Yes"
).astype(int)

df["Churn_enc"] = (
    df["Churn"] == "Yes"
).astype(int)

# One-hot encoding for nominal categorical variables
payment_dummies = pd.get_dummies(
    df["Payment_Method"],
    prefix="Payment",
    drop_first=False,
).astype(int)

region_dummies = pd.get_dummies(
    df["Region"],
    prefix="Region",
    drop_first=False,
).astype(int)

df_encoded = pd.concat(
    [df, payment_dummies, region_dummies],
    axis=1,
)

print("Previous_Campaign_Response -> binary encoded (Yes=1, No=0)")
print("Churn -> binary encoded (Yes=1, No=0), target ready for modelling")

print(
    f"Payment_Method -> one-hot encoded into "
    f"{payment_dummies.shape[1]} columns: "
    f"{list(payment_dummies.columns)}"
)

print(
    f"Region -> one-hot encoded into "
    f"{region_dummies.shape[1]} columns"
)

# 10. FEATURE ENGINEERING

title = "10. FEATURE ENGINEERING"
print(f"\n{title:=^80}\n")

# Feature 1: Share of monthly income spent
df_encoded["Spend_to_Income_Ratio"] = (
    df_encoded["Monthly_Spending"]
    / df_encoded["Monthly_Income"]
).round(3)

# Feature 2: Average spending per purchase
df_encoded["Avg_Spend_per_Purchase"] = (
    df_encoded["Monthly_Spending"]
    / df_encoded["Number_of_Purchases"]
).round(2)

# Feature 3: Purchase efficiency relative to website visits
df_encoded["Visit_to_Purchase_Ratio"] = (
    df_encoded["Number_of_Purchases"]
    / df_encoded["Website_Visits"]
).round(3)

# Feature 4: Flag customers with unusually high support calls
support_75th = df_encoded["Support_Calls"].quantile(0.75)

df_encoded["High_Support_Calls_Flag"] = (
    df_encoded["Support_Calls"] > support_75th
).astype(int)

print("1. Spend_to_Income_Ratio = Monthly_Spending / Monthly_Income")
print("2. Avg_Spend_per_Purchase = Monthly_Spending / Number_of_Purchases")
print("3. Visit_to_Purchase_Ratio = Number_of_Purchases / Website_Visits")
print(
    f"4. High_Support_Calls_Flag = 1 if Support_Calls > "
    f"75th percentile ({support_75th:.0f}) else 0"
)

# The four engineered features above are included in the final modelling-ready dataset.

# 11. FINAL DATA DICTIONARY

title = "11. FINAL DATA DICTIONARY"
print(f"\n{title:=^80}\n")

data_dictionary = [
    ("Customer_ID", "Identifier", "Retained",
     "Unique customer ID; duplicates removed"),

    ("Age", "Numerical", "Retained",
     "No cleaning required; no missing/invalid values"),

    ("Monthly_Income", "Numerical", "Cleaned",
     "Median-imputed, outliers capped at IQR bound"),

    ("Monthly_Spending", "Numerical", "Cleaned",
     "Median-imputed, outliers capped at IQR bound"),

    ("Number_of_Purchases", "Numerical", "Retained",
     "No missing values"),

    ("Website_Visits", "Numerical", "Retained",
     "No missing values"),

    ("Support_Calls", "Numerical", "Retained",
     "No missing values"),

    ("Payment_Method", "Categorical", "Encoded",
     "Mode-imputed, one-hot encoded"),

    ("Region", "Categorical", "Encoded",
     "One-hot encoded (10 regions)"),

    ("Previous_Campaign_Response", "Categorical/Binary", "Encoded",
     "Binary encoded 0/1"),

    ("Churn", "Target/Binary", "Encoded",
     "Binary encoded 0/1 (target variable)"),

    ("Spend_to_Income_Ratio", "Engineered (Numerical)", "New",
     "Monthly_Spending / Monthly_Income"),

    ("Avg_Spend_per_Purchase", "Engineered (Numerical)", "New",
     "Monthly_Spending / Number_of_Purchases"),

    ("Visit_to_Purchase_Ratio", "Engineered (Numerical)", "New",
     "Number_of_Purchases / Website_Visits"),

    ("High_Support_Calls_Flag", "Engineered (Binary)", "New",
     "1 if Support_Calls above 75th percentile"),
]

dd_df = pd.DataFrame(
    data_dictionary,
    columns=[
        "Variable",
        "Type",
        "Status",
        "Preparation performed",
    ],
)

print(dd_df.to_string(index=False))

# MODELLING-READY DATASET

title = "MODELLING-READY DATASET"
print(f"\n{title:=^80}\n")

# Keep original numeric variables, encoded categorical variables engineered features, and the encoded target.
final_cols = (
    [
        "Customer_ID",
        "Age",
        "Monthly_Income",
        "Monthly_Spending",
        "Number_of_Purchases",
        "Website_Visits",
        "Support_Calls",
    ]
    + list(payment_dummies.columns)
    + list(region_dummies.columns)
    + [
        "Previous_Campaign_Response_enc",
        "Spend_to_Income_Ratio",
        "Avg_Spend_per_Purchase",
        "Visit_to_Purchase_Ratio",
        "High_Support_Calls_Flag",
        "Churn_enc",
    ]
)

final_df = df_encoded[final_cols].rename(
    columns={
        "Churn_enc": "Churn",
        "Previous_Campaign_Response_enc": "Previous_Campaign_Response",
    }
)

# Save the final modelling-ready dataset and data dictionary
final_df.to_csv(
    "retail_customer_dataset_modelling_ready.csv",
    index=False,
)

dd_df.to_csv(
    "data_dictionary.csv",
    index=False,
)

print(
    "\nModelling-ready dataset saved: "
    "retail_customer_dataset_modelling_ready.csv "
    f"({final_df.shape[0]} rows x {final_df.shape[1]} columns)"
)
