import joblib
import json

# Load the saved model and TF-IDF vectorizer
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Load the label map from the JSON file
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse the label map for prediction (index -> label)
# The original label_map is {"label": index}, so we reverse it to {index: label}
label_map_reversed = {v: k for k, v in label_map.items()}

# Function to predict the label for new text
def predict_mental_health(text):
    # Preprocess the input text using the TF-IDF vectorizer
    text_tfidf = tfidf.transform([text])
    # Make a prediction
    prediction = model.predict(text_tfidf)
    # Map the predicted label index to the actual label
    predicted_label = label_map_reversed[prediction[0]]
    return predicted_label

# Test the model with some example text
test_texts = [
    "I feel so empty inside and can't get out of bed.",
    "I can't stop worrying about everything.",
    "I keep having flashbacks of the accident.",
    "I can't focus on anything and feel so restless.",
    "I hear voices that aren't there.",
    "I had a great day today and feel happy."
]

# Predict and display results
for text in test_texts:
    prediction = predict_mental_health(text)
    print(f"Text: {text}")
    print(f"Predicted Label: {prediction}")
    print("-" * 50)