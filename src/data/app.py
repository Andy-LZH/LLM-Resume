
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Specify the paths to the joblib files within the Models folder
model_path = 'Models/svm_model.joblib'
vectorizer_path = 'Models/tfidf_vectorizer.joblib'

# Load the pre-trained model and vectorizer
classifier = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        resume_text = request.form['resume_text']

        # Vectorize the input text
        resume_tfidf = vectorizer.transform([resume_text])

        # Make a prediction
        prediction = classifier.predict(resume_tfidf)[0]

        return render_template('index.html', resume_text=resume_text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)