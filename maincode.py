### Marketing Campaign Analysis: analysing customer behaviour and campaign response

#==========================================================================
## 1) Importing Required Libraries
#==========================================================================
# Install the pinned versions once before running:
#   pip install -r requirements.txt

# Set Visualisation Style
seaborn_style = 'whitegrid'

# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import math
from sklearn.linear_model import LogisticRegression 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# Apply seaborn style
sns.set_style(seaborn_style)

# Load the CSV file
file_path = "marketing_campaign.csv"
df = pd.read_csv(file_path, sep=';')

#===========================================================================
## 2) Basic Data Exploration, Visualisation & Basic Data Cleaning 
#===========================================================================

# View First 5 Rows of Dataset 
print("🔍 First 5 rows of the dataset:\n")                               
print(df.head())

# Get Dataset Info
print("\n📊 Dataset Info:\n")                                           
print(df.info())

# Check for Missing Values
print("\n❓ Missing Values:\n")                                           
print(df.isnull().sum())

# Remove the rows where there is at least 1 missing value 
df_cleaned = df.dropna().copy()  # .copy() so later edits don't act on a view of df

# Optional: Check how many rows were removed
print(f"🧹 Rows before cleaning: {df.shape[0]}")
print(f"🧼 Rows after cleaning:  {df_cleaned.shape[0]}")

# Descriptive Statistics
print("\n📈 Descriptive Statistics:\n")                                  
print(df_cleaned.describe())

# Plot Correlation Heatmap
plt.figure(figsize=(12,6))                                                
plt.title('Correlation Heatmap of Marketing Campaign Data')
sns.heatmap(df_cleaned.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Plot Distribution of Numerical Features
# Get Numerical Columns 
numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns   

# Filter out binary columns and 'ID'
numeric_cols_filtered = [
    col for col in numerical_cols 
    if df_cleaned[col].nunique() > 2 and col.lower() != 'id'
]

# Set up subplot grid
num_cols = len(numeric_cols_filtered)
cols = 3  # Number of columns in subplot grid
rows = math.ceil(num_cols / cols)

# Plot all distributions in one figure
plt.figure(figsize=(cols * 5, rows * 4))

for i, col in enumerate(numeric_cols_filtered, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(df_cleaned[col], kde=True, bins=30)
    plt.title(f"{col}")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.suptitle("📊 Distribution of Numerical Features (Excluding Binary & ID)", fontsize=16, y=1.02)
plt.show()

# Plot Categorical Columns 
# Identify Categorical Columns (excluding 'Dt_Customer')
categorical_cols = [col for col in df_cleaned.select_dtypes(include='object').columns if col != 'Dt_Customer']

# Set up subplot grid layout
num_cols = len(categorical_cols)
cols = 2  # Number of columns in the grid
rows = math.ceil(num_cols / cols)

plt.figure(figsize=(cols * 5, rows * 4))

# Plot Count of Each Categorical Column
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(rows, cols, i)
    sns.countplot(data=df_cleaned, x=col, order=df_cleaned[col].value_counts().index)
    plt.title(f"{col}")
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.suptitle("📋 Count of Categorical Features (Excl. Dt_Customer)", fontsize=16, y=1.02)
plt.show()

#===========================================================================
## 3) Data Cleaning & Data Transformation
#===========================================================================

# Check how many duplicate rows exist
num_duplicates = df_cleaned.duplicated().sum()
print(f"🔍 Number of duplicate rows in the dataset: {num_duplicates}")  # No Duplicates , No Further updates needed

# 'YOLO', 'Alone', and 'Absurd' are likely non-serious or inconsistent entries,
# so we'll group them under 'Single' for meaningful analysis.
print("🔍 Unique values in 'Marital_Status':")
print(df_cleaned['Marital_Status'].unique())

print("\n📊 Value counts in 'Marital_Status':")      
print(df_cleaned['Marital_Status'].value_counts())   

# Replace unusual values with 'Single'
df_cleaned.loc[:, 'Marital_Status'] = df_cleaned['Marital_Status'].replace({
    'YOLO': 'Single',
    'Absurd': 'Single',
    'Alone': 'Single'
})

# Verify the replacement
print("🔎 Unique values in 'Marital_Status' after replacement:")
print(df_cleaned['Marital_Status'].unique())
print("\n📊 Value counts in 'Marital_Status' after replacement:")
print(df_cleaned['Marital_Status'].value_counts())

# Check for Outliers through Visualisation
# Columns to visualise and detect outliers
num_vars = [
    'Income','Year_Birth', 'Kidhome', 'Teenhome', 'Recency', 
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
    'NumWebVisitsMonth'
]

# Visualise outliers using boxplots
plt.figure(figsize=(18, 20))

for i, col in enumerate(num_vars, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=df_cleaned[col], color='skyblue')
    plt.title(col)
    plt.xlabel("")  # Optional: clean layout

plt.tight_layout()
plt.suptitle("📦 Boxplots for Outlier Detection", fontsize=18, y=1.02)
plt.show()

# Removing outliers from Income using IQR
Q1 = df_cleaned['Income'].quantile(0.25)
Q3 = df_cleaned['Income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df_cleaned[(df_cleaned['Income'] >= lower_bound) & (df_cleaned['Income'] <= upper_bound)]

# Optional: Check new Income distribution
plt.figure(figsize=(4, 4))
sns.boxplot(y=df_cleaned['Income'], color='skyblue')
plt.title("Income After Outlier Removal")
plt.tight_layout()
plt.show()

# Removing outliers from Year_Birth using IQR
Q1 = df_cleaned['Year_Birth'].quantile(0.25)
Q3 = df_cleaned['Year_Birth'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df_cleaned[(df_cleaned['Year_Birth'] >= lower_bound) & (df_cleaned['Year_Birth'] <= upper_bound)]

# Optional: Check new Year_Birth distribution
plt.figure(figsize=(4, 4))
sns.boxplot(y=df_cleaned['Year_Birth'], color='skyblue')
plt.title("Year_Birth After Outlier Removal")
plt.tight_layout()
plt.show()

# Print confirmation
print(f"✅ Shape of dataset after removing outliers in 'Income' and 'Year_Birth': {df_cleaned.shape}")


# Check data type of 'Dt_Customer' and a few of its values 
print("\n🔍 Data type of 'Dt_Customer':")
print(df_cleaned['Dt_Customer'].dtype)
print(df_cleaned['Dt_Customer'].head())
# Convert 'Dt_Customer' to datetime format from object string type
## Plain column assignment (not .loc[:, ...]) so the column becomes a true datetime dtype
df_cleaned['Dt_Customer'] = pd.to_datetime(df_cleaned['Dt_Customer'], format='%Y-%m-%d')
print(f"📅 Customers enrolled between {df_cleaned['Dt_Customer'].min().date()} "
      f"and {df_cleaned['Dt_Customer'].max().date()}")


# Total Campaigns Accepted Transformation
# Creating a new column to sum the accepted campaigns 
df_cleaned.loc[:, 'Total_Campaigns_Accepted'] = df_cleaned[[
    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
    'AcceptedCmp4', 'AcceptedCmp5', 'Response'
]].sum(axis=1)

# Age Group Transformation
## Create Age column
## Age is measured at the time the data was collected (the latest enrolment year),
## not today's date - otherwise every customer looks about 11 years older than they were
reference_year = df_cleaned['Dt_Customer'].dt.year.max()
print(f"📅 Ages measured as at {reference_year}")
df_cleaned.loc[:, 'Age'] = reference_year - df_cleaned['Year_Birth']
# Create Age Group column using pd.cut
df_cleaned.loc[:, 'Age_Group'] = pd.cut(df_cleaned['Age'],
                                        bins=[0, 30, 45, 60, 100],
                                        labels=['Young', 'Adult', 'Middle-Aged', 'Senior'])

# Create a binary column : 1 if accepted any campaign, 0 otherwise
df_cleaned['Responded'] = (df_cleaned['Total_Campaigns_Accepted'] > 0).astype(int)

#===========================================================================
## 4) Preliminary Analysis and visualisation
#===========================================================================

# Compare Total Purchases by Channel 
channel_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df_cleaned[channel_cols].sum().sort_values(ascending=False).plot(kind='bar')
plt.title("Total Purchases by Marketing Channel")
plt.ylabel("Number of Purchases")
plt.show()

# Campaign Acceptance vs Channel usage - Check if people who accepted campaigns tend to use certain channels more. 
# Create a mask for customers who accepted at least one campaign
responders = df_cleaned[df_cleaned['Total_Campaigns_Accepted'] > 0]

# Average purchases per channel by responders
responders[channel_cols].mean().plot(kind='bar', color='green')
plt.title("Average Channel Purchases for Campaign Responders")
plt.ylabel("Average Purchases")
plt.show()

# Cross-tab by Age - Which channel works better for which demographic? 
# Average web purchases by age group
df_cleaned.groupby('Age_Group', observed=True)['NumWebPurchases'].mean().plot(kind='bar')
plt.title("Avg Web Purchases by Age Group")
plt.ylabel("Average")
plt.show()

# Define product columns
product_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

# ===========================================================================  
## 5) What product categories are most popular ? : Question 1 
# ===========================================================================

# Calculate % of customers who purchased each product (non-zero spend) 
## This Reflects the *reach* - how widely each product category is adopted 
popular_counts = (df_cleaned[product_cols] > 0).sum().sort_values(ascending=False)

# Calculate average spend per customer per category
## This reflects the *depth* - how much customers typically spend on each category
avg_spend = df_cleaned[product_cols].mean().sort_values(ascending=False)

# Combine both metrics into single data frame for a fairer story
combined = pd.DataFrame({
    '% Customers': popular_counts / len(df_cleaned),
    'Avg Spend': avg_spend
})

# Normalise the two columns to 0–1 scale
## This is necessary to fairly combine reach and spend into a score 
combined['% Customers (Norm)'] = combined['% Customers'] / combined['% Customers'].max()
combined['Avg Spend (Norm)'] = combined['Avg Spend'] / combined['Avg Spend'].max()

# Create a composite popularity score (equal weighting)
## 50% weight of % of customers (reach) and 50% to average spend (depth)
combined['Popularity Score'] = 0.5 * combined['% Customers (Norm)'] + 0.5 * combined['Avg Spend (Norm)']

# Sort by the final score
combined_sorted = combined.sort_values(by='Popularity Score', ascending=False)

# Round values for cleaner display and display table 
display_table = combined_sorted[['% Customers', 'Avg Spend', 'Popularity Score']].round(3)
print(display_table)

# Plot popularity scores
plt.figure(figsize=(10, 6))
sns.barplot(
    x=display_table['Popularity Score'],
    y=display_table.index,
    hue=display_table.index,
    palette='Blues_d',
    legend=False
)
plt.title('Product Category Popularity (Composite Score)')
plt.xlabel('Popularity Score')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()

# ===========================================================================  
## 6) Which Customer Segments are most likely to respond to campaigns? : Question 2 
# ===========================================================================

# "Responded" = accepted at least one of the six campaigns (AcceptedCmp1-5 and Response)
# The overall rate is the baseline each segment is compared against
overall_rate = df_cleaned['Responded'].mean()
print(f"✅ Overall response rate: {overall_rate:.3f} "
      f"({df_cleaned['Responded'].sum()} of {len(df_cleaned)} customers)")

def segment_response(col, palette, title):
    """Response rate by segment, shown with each group's size and a chi-square test.

    Group size matters: a rate from a handful of customers can top the ranking by chance.
    The chi-square test asks whether response rate varies across the groups at all.
    """
    summary = (df_cleaned.groupby(col, observed=True)['Responded']
               .agg(Customers='size', Responders='sum', Rate='mean')
               .sort_values('Rate', ascending=False))
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df_cleaned[col], df_cleaned['Responded']))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=summary.index.astype(str), y=summary['Rate'].values,
                hue=summary.index.astype(str), palette=palette, legend=False)
    plt.axhline(overall_rate, color='black', linestyle='--', linewidth=1, label='Overall rate')
    plt.title(title)
    plt.ylabel("Response Rate")
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    print(f"\n✅ Response Rate by {col}:")
    print(summary.round(3))
    print(f"   Chi-square test across groups: chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4f}")
    return summary

def one_vs_rest(col, value):
    """Chi-square test of one group against everyone else (e.g. PhD vs all other levels).

    The group tested is the top-ranked one, chosen after seeing the data, so treat the p-value
    as somewhat optimistic.
    """
    p = stats.chi2_contingency(pd.crosstab(df_cleaned[col] == value, df_cleaned['Responded']))[1]
    print(f"   {value} vs all other customers: p = {p:.4f}")

# Response Rate by Age Group (ages as at the reference year)
response_by_age = segment_response('Age_Group', "viridis", "📊 Campaign Response Rate by Age Group")

# Response Rate by Education Level
response_by_edu = segment_response('Education', "magma", "📊 Campaign Response Rate by Education Level")
one_vs_rest('Education', response_by_edu.index[0])

# Response Rate by Marital Status
response_by_marital = segment_response('Marital_Status', "cubehelix", "📊 Campaign Response Rate by Marital Status")
one_vs_rest('Marital_Status', response_by_marital.index[0])

# Customers living with a partner (Married, Together) against those who are not (Single, Divorced, Widow)
has_partner = df_cleaned['Marital_Status'].isin(['Married', 'Together'])
partner_table = df_cleaned.groupby(has_partner)['Responded'].agg(Customers='size', Responders='sum', Rate='mean')
partner_table.index = partner_table.index.map({True: 'With partner', False: 'Without partner'})
partner_table.index.name = 'Partner'
print("\n✅ Response Rate by Partner status:")
print(partner_table.round(3))
p_partner = stats.chi2_contingency(pd.crosstab(has_partner, df_cleaned['Responded']))[1]
print(f"   Without partner vs with partner: p = {p_partner:.4f}")



# ===========================================================================  
## 7) Logistic Regression: Which Channel Predicts Campaign Response : Question 3 
# ===========================================================================

# Average purchases per channel, responders vs non-responders (no other channel held constant)
print("\n✅ Average purchases per channel by response:")
print(df_cleaned.groupby('Responded')[channel_cols].mean().round(2))

# Select Features (x) and Target (y)
X = df_cleaned[channel_cols]  # already defined as ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
y = df_cleaned['Responded']

# Standardise the features for interpretability
## Note: the scaler is fitted on all rows before the train/test split (minor leakage into the test set);
## kept as originally published so the coefficients stay comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (70% train / 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Add intercept to the features 
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit the logistic regression model on train data
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Print detailed summary of the model
print(result.summary())

# Predict on test set
y_pred_prob = result.predict(X_test_const)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate model
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

auc = roc_auc_score(y_test, y_pred_prob)
print(f"\n🔎 ROC-AUC Score: {auc:.3f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve - Campaign Response Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Visualise the influence of each channel on campaign response
# Create a DataFrame for coefficients (excluding intercept)
coef_df = pd.DataFrame({
    'Channel': channel_cols,
    'Coefficient': result.params[1:]  # skip intercept
})

# Sort channels by coefficient (helps readability)
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

# Plot
plt.figure(figsize=(7, 4))
ax = sns.barplot(
    data=coef_df, 
    x='Coefficient', 
    y='Channel', 
    hue='Channel',
    palette='coolwarm', 
    legend=False,
    edgecolor='black'
)

# Add vertical line at 0
plt.axvline(0, color='black', linestyle='--', linewidth=1)

# Add annotations on bars
for i, (coef, channel) in enumerate(zip(coef_df['Coefficient'], coef_df['Channel'])):
    ax.text(
        coef, i, f"{coef:.2f}", 
        va='center', ha='left' if coef > 0 else 'right', 
        fontsize=9, color="black", fontweight="bold"
    )

# Titles and labels
plt.title("Channel Influence on Campaign Acceptance", fontsize=13, fontweight="bold")
plt.xlabel("Logistic Regression Coefficient", fontsize=11)
plt.ylabel("Channel", fontsize=11)

# Clean style
sns.despine(left=True, bottom=True)
plt.grid(axis='x', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Print ranked influence of channels
print("📊 Channel Influence on Campaign Response (by Coefficient):")
print(coef_df)
