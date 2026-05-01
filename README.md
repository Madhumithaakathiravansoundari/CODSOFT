# Machine Learning Internship Projects – CodSoft

This repository contains four machine learning projects completed as part of my Machine Learning Internship at CodSoft. These projects focus on solving real-world classification problems using Python, Scikit-learn, and data preprocessing techniques.

---

# Task 1: Movie Genre Classification

## Objective

Build a machine learning model that predicts the genre of a movie based on its plot summary using Natural Language Processing (NLP).

## Technologies Used

* Python
* Pandas
* TF-IDF Vectorization
* Logistic Regression
* Support Vector Machine (SVM)
* Scikit-learn

## Workflow

* Loaded training and testing datasets
* Preprocessed textual movie plot summaries
* Converted text into numerical vectors using TF-IDF
* Trained models using Logistic Regression and LinearSVC
* Evaluated model accuracy
* Saved predictions into a CSV file

## Output

The model predicts movie genres based on plot descriptions and stores the results in `predicted_genres.csv`.

---

# Task 2: Credit Card Fraud Detection

## Objective

Develop a machine learning model to detect fraudulent credit card transactions using transaction details and customer-related features.

## Technologies Used

* Python
* Pandas
* Label Encoding
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Scikit-learn

## Workflow

* Loaded fraud transaction datasets
* Selected important features such as category, amount, gender, city population, and job
* Applied Label Encoding to categorical columns
* Trained Logistic Regression, Decision Tree, and Random Forest models
* Compared model accuracy
* Exported prediction results into a CSV file

## Output

The final predictions are stored in `fraud_predictions.csv`.

---

# Task 3: Customer Churn Prediction

## Objective

Build a model to predict customer churn for a subscription-based service using customer demographics and banking behavior.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Label Encoding
* Logistic Regression
* Random Forest Classifier
* Scikit-learn

## Workflow

* Loaded and explored the customer churn dataset
* Removed unnecessary columns such as CustomerId and Surname
* Converted categorical variables using Label Encoding
* Split the dataset into training and testing sets
* Trained Logistic Regression and Random Forest models
* Evaluated using Accuracy Score, Classification Report, and Confusion Matrix
* Visualized feature importance using Random Forest

## Output

The model predicts whether a customer is likely to leave the service or remain.

---

# Task 4: SMS Spam Detection

## Objective

Create a machine learning model to classify SMS messages as Spam or Legitimate (Ham) using NLP techniques.

## Technologies Used

* Python
* Pandas
* TF-IDF Vectorization
* Multinomial Naive Bayes
* Scikit-learn

## Workflow

* Loaded the SMS spam dataset
* Selected label and message columns
* Converted labels (Ham/Spam) into numerical values
* Applied TF-IDF Vectorization for feature extraction
* Split the dataset into training and testing sets
* Trained the model using Multinomial Naive Bayes
* Evaluated performance using Accuracy Score, Classification Report, and Confusion Matrix
* Tested with custom SMS messages for prediction

## Output

The model classifies messages as Spam or Legitimate.

---

# Conclusion

These four projects provided practical experience in:

* Natural Language Processing (NLP)
* Classification Algorithms
* Data Preprocessing
* Feature Engineering
* Model Evaluation
* Real-world Machine Learning Applications

This internship helped strengthen my understanding of Machine Learning concepts and their implementation in solving real-world problems.

---

# Author

Machine Learning Internship Project
Completed under CodSoft Internship Program
