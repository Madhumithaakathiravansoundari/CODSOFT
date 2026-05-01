# customerchurn.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv(r"C:\Users\Madhu\Downloads\CODSOFT\Task3\Customer Churn Dataset\Churn_Modelling.csv")

# Display first 5 rows
print("First 5 Rows:\n")
print(data.head())

# Drop unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical columns
le = LabelEncoder()

data["Geography"] = le.fit_transform(data["Geography"])
data["Gender"] = le.fit_transform(data["Gender"])

# Features and Target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# -------------------------------
# Logistic Regression Model
# -------------------------------
print("\n===== Logistic Regression =====")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# -------------------------------
# Random Forest Model
# -------------------------------
print("\n===== Random Forest =====")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# -------------------------------
# Feature Importance Plot
# -------------------------------
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh")

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()