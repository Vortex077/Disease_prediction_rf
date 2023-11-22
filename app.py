from flask import Flask,request,jsonify
import joblib
import zipfile
from joblib import load
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


extract_path = tempfile.mkdtemp()
with zipfile.ZipFile('random_forest_java_model_02.zip') as zip_ref:
    zip_ref.extractall()
rf_model=load("random_forest_java_model_02.pkl")
X_transformed = joblib.load("X_transformed.joblib")
vectorizer = joblib.load("fitted_vectorizer.joblib")

app = Flask(__name__)
@app.route("/")
def home():
    return "Hello World!"
@app.route('/predict',methods=["POST"])
def predict():
    try:
        symptoms = request.form.get("symptoms")

        # Check if symptoms are provided
        if symptoms is None or symptoms == "":
            return jsonify({"error": "Symptoms not provided"}), 400

        # Transform the input using the same vectorizer
        symptoms_transformed = vectorizer.transform([symptoms])

        # Make predictions
        result = rf_model.predict(symptoms_transformed)

        return jsonify({"result": result.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
