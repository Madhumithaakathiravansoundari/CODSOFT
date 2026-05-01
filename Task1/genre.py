import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
train_df = pd.read_csv(
    r"C:\Users\Madhu\Downloads\CODSOFT\Task1\Genre Classification Dataset\train_data.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["id", "title", "genre", "plot"]
)
test_df = pd.read_csv(
    r"C:\Users\Madhu\Downloads\CODSOFT\Task1\Genre Classification Dataset\test_data.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["id", "title", "plot"]
)
test_solution = pd.read_csv(
    r"C:\Users\Madhu\Downloads\CODSOFT\Task1\Genre Classification Dataset\test_data_solution.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["id", "title", "genre" , "plot"]
)
X_train = train_df["plot"]
y_train = train_df["genre"]
X_test = test_df["plot"]
y_test = test_solution["genre"]
tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
lr = LogisticRegression(max_iter=300)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)
print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, lr_preds))
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
svm_preds = svm.predict(X_test_tfidf)
print("SVM Accuracy:")
print(accuracy_score(y_test, svm_preds))
output = pd.DataFrame({
    "id": test_df["id"],
    "title": test_df["title"],
    "true_genre": y_test,
    "lr_predicted": lr_preds,
    "svm_predicted": svm_preds
})
output.to_csv("predicted_genres.csv", index=False)
print("Predictions saved to predicted_genres.csv")
