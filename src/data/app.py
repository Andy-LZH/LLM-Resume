from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import re

app = Flask(__name__)

model_path = 'Models/svm_model.joblib'
vectorizer_path = 'Models/tfidf_vectorizer.joblib'

#getting pretrained model
classifier = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def cleanResume(text):
    # These are what is being substituded
    patterns = [
        r'http\S+\s*',       # remove URLs
        r'RT|cc',            # remove RT and cc
        r'#\S+',             # remove hashtags
        r'@\S+',             # remove mentions
        r'[^\w\s]',          # remove punctuations except underscore
        r'[^\x00-\x7f]'      # remove non-ASCII characters
    ]
    
    # Apply substitutions
    for pattern in patterns:
        text = re.sub(pattern, ' ', text)
    
    # Remove extra whitespace
    text = re.sub('\s+', ' ', text) # Remove leading/trailing spaces
    
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        resume_text = request.form['resume_text']

        # Clean the input text
        cleaned_resume = cleanResume(resume_text)

        #print("Cleaned Input:", cleaned_resume)
        
        # Vectorize the cleaned text
        resume_tfidf = vectorizer.transform([cleaned_resume])

        # Make a prediction
        prediction = classifier.predict(resume_tfidf)[0]

        return render_template('index.html', resume_text=resume_text, cleaned_resume=cleaned_resume, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
