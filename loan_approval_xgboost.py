# =====================================================
# HDFC LOAN APPROVAL SYSTEM - XGBOOST VERSION
# =====================================================

# 1. INSTALL & IMPORTS
!pip install xgboost -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

sns.set_style("whitegrid")

# 2. UPLOAD DATA
print("Please upload 'loan_approval_dataset.csv'")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# 3. LOAD & CLEAN DATA
df_raw = pd.read_csv(filename)
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(" ", "_")

for col in df_raw.select_dtypes(include=['object']).columns:
   df_raw[col] = df_raw[col].str.strip()

if 'loan_id' in df_raw.columns:
   df_raw.drop(columns='loan_id', inplace=True)

def rupees_to_lakhs(x, pos):
   return f"{int(x/100000)}"

# =====================================================
# EDA VISUALIZATIONS
# =====================================================

plt.figure(figsize=(5,4))
sns.countplot(x='loan_status', data=df_raw, palette='viridis')
plt.title("Loan Status Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df_raw["income_annum"], bins=30, kde=True, color='blue')
plt.gca().xaxis.set_major_formatter(FuncFormatter(rupees_to_lakhs))
plt.xlabel("Annual Income (₹ Lakhs)")
plt.title("Applicant Income Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df_raw["loan_amount"], bins=30, kde=True, color='green')
plt.gca().xaxis.set_major_formatter(FuncFormatter(rupees_to_lakhs))
plt.xlabel("Loan Amount (₹ Lakhs)")
plt.title("Loan Amount Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='loan_status', y="cibil_score", data=df_raw, palette='magma')
plt.title("Credit Score vs Loan Status")
plt.show()

# =====================================================
# MODEL TRAINING
# =====================================================

df_model = df_raw.copy()
encoders = {}
cat_cols = ['education', 'self_employed', 'loan_status']

for col in cat_cols:
   le = LabelEncoder()
   df_model[col] = le.fit_transform(df_model[col])
   encoders[col] = le

X = df_model.drop('loan_status', axis=1)
y = df_model['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
   n_estimators=200,
   learning_rate=0.05,
   max_depth=5,
   random_state=42,
   use_label_encoder=False,
   eval_metric='logloss'
)
model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

y_pred = model.predict(X_test)

print("\n" + "="*30)
print(f"MODEL ACCURACY: {round(accuracy_score(y_test, y_pred)*100, 2)}%")
print("="*30)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoders['loan_status'].classes_))

plt.figure(figsize=(5,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
           xticklabels=encoders['loan_status'].classes_,
           yticklabels=encoders['loan_status'].classes_)
plt.title("Confusion Matrix (XGBoost)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fi = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=fi)
plt.title("Top Factors Affecting Loan Approval (XGBoost)")
plt.show()

# =====================================================
# PREDICTION
# =====================================================

print("\n" + "="*30)
print("ENTER APPLICANT DETAILS FOR PREDICTION")
print("="*30)

user_data = {}
for col in X.columns:
   if col in encoders:
       options = encoders[col].classes_
       val = input(f"Enter {col} ({'/'.join(options)}): ").strip()
       user_data[col] = encoders[col].transform([val])[0]
   else:
       user_data[col] = float(input(f"Enter {col}: "))

user_df = pd.DataFrame([user_data])
pred_num = model.predict(user_df)[0]
final_decision = encoders['loan_status'].inverse_transform([pred_num])[0]

print(f"\n>>> FINAL LOAN DECISION: {final_decision.upper()} <<<")
