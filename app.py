from flask import Flask, render_template, request
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Home Page
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        result = model.predict(vector)[0]

        print("Model predicted:", result)

        if result == "FAKE":
            prediction = "FAKE NEWS"
        else:
            prediction = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=prediction
    )

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)