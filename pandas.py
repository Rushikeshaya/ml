import numpy as np
import pandas as pd

# ==============================================================================
# 1. CREATE A REALISTIC RAW ML DATASET
# ==============================================================================

raw_data = {
    "CustomerID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Signup_Date": [
        "2023-01-15",
        "2023-02-20",
        "2023-03-05",
        "2023-03-22",
        "2023-04-10",
        "2023-05-01",
        "2023-05-18",
        "2023-06-01",
    ],
    "Age": [22, 45, np.nan, 35, 60, 28, 85, 31],  # Contains NaN & Outlier (85)
    "Income": [
        45000,
        85000,
        62000,
        np.nan,
        120000,
        50000,
        150000,
        58000,
    ],  # Contains NaN
    "Education": [
        "High School",
        "Bachelor",
        "Master",
        "Bachelor",
        "PhD",
        "High School",
        "Master",
        "Bachelor",
    ],
    "Device": [
        "Mobile",
        "Desktop",
        "Mobile",
        "Tablet",
        "Desktop",
        "Mobile",
        "Desktop",
        "Mobile",
    ],
    "Email": [
        "user1@gmail.com",
        "user2@yahoo.com",
        "user3@gmail.com",
        "user4@company.org",
        "user5@gmail.com",
        "user6@yahoo.com",
        "user7@company.org",
        "user8@gmail.com",
    ],
    "Purchased": [0, 1, 0, 1, 1, 0, 1, 0],  # Target variable (y)
}

df = pd.DataFrame(raw_data)
print("--- Raw Dataset ---")
print(df)


# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA) FOR ML
# ==============================================================================

# Statistical summary of numerical columns (check distributions, min, max, percentiles)
print("\n--- Numerical Summary ---")
print(df.describe())

# Check target distribution (crucial to detect class imbalance)
print("\n--- Target Class Balance (Ratios) ---")
print(df["Purchased"].value_counts(normalize=True))

# Check unique values per column to identify high-cardinality features
print("\n--- Unique Values Count ---")
print(df.nunique())


# ==============================================================================
# 3. HANDLING MISSING DATA (IMPUTATION)
# ==============================================================================

# For skewed numerical columns or data with outliers, use Median instead of Mean
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)

income_median = df["Income"].median()
df["Income"] = df["Income"].fillna(income_median)

# Optional ML technique: Add a binary indicator flag showing that a value was missing
# df['Age_was_missing'] = df['Age'].isna().astype(int)


# ==============================================================================
# 4. OUTLIER HANDLING (CLIPPING)
# ==============================================================================

# Cap extreme values at specific percentiles instead of dropping rows
lower_cap = df["Age"].quantile(0.05)
upper_cap = df["Age"].quantile(0.95)

# Clip restricts all values within [lower_cap, upper_cap]
df["Age_Clipped"] = df["Age"].clip(lower=lower_cap, upper=upper_cap)


# ==============================================================================
# 5. FEATURE ENGINEERING
# ==============================================================================

# --- A. Date / Time Extraction ---
# Convert string to datetime format
df["Signup_Date"] = pd.to_datetime(df["Signup_Date"])

# Extract cyclical & linear time components
df["Signup_Year"] = df["Signup_Date"].dt.year
df["Signup_Month"] = df["Signup_Date"].dt.month
df["Signup_DayOfWeek"] = (
    df["Signup_Date"].dt.dayofweek
)  # Monday = 0, Sunday = 6
df["Is_Weekend"] = df["Signup_DayOfWeek"].isin([5, 6]).astype(int)

# --- B. String / Text Parsing ---
# Extract email domain provider from email address
df["Email_Domain"] = df["Email"].str.split("@").str[1]

# --- C. Numerical Binning / Discretization ---
# pd.cut(): Divides data into predefined custom buckets
bins = [0, 30, 50, 100]
labels = ["Young", "Middle", "Senior"]
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

# --- D. Mathematical Transformations ---
# Log transformation to reduce skewness in monetary features (np.log1p avoids log(0))
df["Income_Log"] = np.log1p(df["Income"])


# ==============================================================================
# 6. ENCODING CATEGORICAL VARIABLES
# ==============================================================================

# --- A. Ordinal Encoding (For features with a meaningful hierarchy) ---
education_mapping = {
    "High School": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4,
}
df["Education_Encoded"] = df["Education"].map(education_mapping)

# --- B. One-Hot Encoding (For nominal features with no inherent order) ---
# drop_first=True helps avoid multicollinearity (the "dummy variable trap")
df = pd.get_dummies(
    df, columns=["Device", "Email_Domain", "Age_Group"], drop_first=True
)


# ==============================================================================
# 7. FEATURE SELECTION & PREPARING X and y FOR SCIKIT-LEARN
# ==============================================================================

# Drop unique identifier columns and raw pre-transformed columns
columns_to_drop = [
    "CustomerID",  # Random ID (causes overfitting)
    "Signup_Date",  # Replaced by extracted date features
    "Email",  # Replaced by Email_Domain
    "Education",  # Replaced by Education_Encoded
    "Age",  # Replaced by Age_Clipped
    "Income",  # Replaced by Income_Log
    "Purchased",  # Target column
]

# Separate features (X) and target (y)
X = df.drop(columns=columns_to_drop)
y = df["Purchased"]

print("\n--- Final ML Feature Matrix (X) ---")
print(X.head())

print("\n--- Target Vector (y) ---")
print(y.head())

# Convert boolean columns to integers (0 and 1) for ML models
X = X.astype(float)


# ==============================================================================
# 8. CORRELATION MATRIX (FEATURE FILTERING)
# ==============================================================================

# Calculate Pearson correlation to see feature relationships with target
full_processed_df = pd.concat([X, y], axis=1)
correlation_with_target = (
    full_processed_df.corr()["Purchased"].sort_values(ascending=False)
)

print("\n--- Feature Correlation with Target ('Purchased') ---")
print(correlation_with_target)