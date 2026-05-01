## Dataset
Source:https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv("Dataset.csv", encoding="latin-1")
data = data[["v1", "v2"]]
data.columns = ["Label", "Message"]
data["Label"] = data["Label"].map({"ham": 0, "spam": 1})
print("First 5 Rows:\n")
print(data.head())
X = data["Message"]
y = data["Label"]
tfidf = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n===== Spam Detection Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
sample = ["Congratulations! You won a free mobile recharge. Call now!"]
sample_tfidf = tfidf.transform(sample)
result = model.predict(sample_tfidf)
print("\nCustom Message Prediction:")
if result[0] == 1:
    print("Spam Message")
else:
    print("Legitimate (Ham) Message")
