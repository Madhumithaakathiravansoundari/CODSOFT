import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
train_df = pd.read_csv(
    r"C:\Users\Madhu\Downloads\CODSOFT\Task2\Fraud Detection\fraudTrain.csv"
)
test_df = pd.read_csv(
    r"C:\Users\Madhu\Downloads\CODSOFT\Task2\Fraud Detection\fraudTest.csv"
)
print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
columns = [
    "category",
    "amt",
    "gender",
    "city_pop",
    "job",
    "is_fraud"
]
train_df = train_df[columns]
test_df = test_df[columns]
le = LabelEncoder()
for col in ["category", "gender", "job"]:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.fit_transform(test_df[col])
X_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]
X_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, dt_pred))
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))
output = pd.DataFrame({
    "Actual": y_test,
    "Logistic_Regression": lr_pred,
    "Decision_Tree": dt_pred,
    "Random_Forest": rf_pred
})
output.to_csv("fraud_predictions.csv", index=False)
print("\nPredictions saved to fraud_predictions.csv")
