from flask import Flask, request, jsonify, render_template
import joblib
import json

app = Flask(__name__)

# Load the saved model and TF-IDF vectorizer
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Load the label map from the JSON file
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse the label map for prediction (index -> label)
label_map_reversed = {v: k for k, v in label_map.items()}

# Function to predict the label and confidence for a message
def predict_mental_health_with_confidence(text):
    # Preprocess the input text using the TF-IDF vectorizer
    text_tfidf = tfidf.transform([text])
    # Make a prediction
    prediction = model.predict(text_tfidf)
    # Get confidence scores (probabilities) for each class
    confidence_scores = model.predict_proba(text_tfidf)[0]
    # Map the predicted label index to the actual label
    predicted_label = label_map_reversed[prediction[0]]
    # Create a dictionary of confidence scores for each label
    confidence_dict = {label_map_reversed[i]: f"{score * 100:.2f}%" for i, score in enumerate(confidence_scores)}
    
    # Find the highest confidence
    max_confidence_label = max(confidence_dict, key=lambda k: confidence_dict[k])
    max_confidence_value = confidence_dict[max_confidence_label]

    # If the highest confidence is above a threshold (e.g., 80%), suggest contacting
    suggestion = ""
    if float(max_confidence_value.replace('%', '')) > 80:
        suggestion = "We recommend contacting v-chance@vitap.ac.in for further assistance."

    return predicted_label, confidence_dict, suggestion

# Route for the home page (chat interface)
@app.route("/")
def home():
    return render_template("chat.html")

# Route to handle message predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["message"]
    predicted_label, confidence_dict, suggestion = predict_mental_health_with_confidence(text)
    return jsonify({
        "predicted_label": predicted_label,
        "confidence_scores": confidence_dict,
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(debug=True)
