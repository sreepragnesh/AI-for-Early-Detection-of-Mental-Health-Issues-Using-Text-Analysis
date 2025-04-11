import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
dataset = pd.read_csv("large_mental_health_dataset.csv")

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

# Convert labels to numerical values
label_map = {label: idx for idx, label in enumerate(train_df["label"].unique())}
train_df["label"] = train_df["label"].map(label_map)
test_df["label"] = test_df["label"].map(label_map)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train = tfidf.fit_transform(train_df["text"])
X_test = tfidf.transform(test_df["text"])

# Labels
y_train = train_df["label"]
y_test = test_df["label"]

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save the model and TF-IDF vectorizer
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

import json

# Save label map to a JSON file
with open("label_map.json", "w") as f:
    json.dump(label_map, f)