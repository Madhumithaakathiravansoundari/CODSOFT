# spam.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv(r"Spam Dataset\spam.csv", encoding="latin-1")

# Keep only needed columns
data = data[["v1", "v2"]]

# Rename columns
data.columns = ["Label", "Message"]

# Convert labels: ham = 0, spam = 1
data["Label"] = data["Label"].map({"ham": 0, "spam": 1})

print("First 5 Rows:\n")
print(data.head())

# Features and Target
X = data["Message"]
y = data["Label"]

# Convert text into numerical values using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")

X_tfidf = tfidf.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\n===== Spam Detection Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Test with custom message
sample = ["Congratulations! You won a free mobile recharge. Call now!"]

sample_tfidf = tfidf.transform(sample)
result = model.predict(sample_tfidf)

print("\nCustom Message Prediction:")
if result[0] == 1:
    print("Spam Message")
else:
    print("Legitimate (Ham) Message")