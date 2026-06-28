import pandas as pd
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("fake_news.csv")

# Check labels
print(data["label"].value_counts())

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Clean dataset
data["text"] = data["text"].apply(clean_text)

# Features and Labels
X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_accuracy = 0
best_model = None

# Train Models
for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("Accuracy:", acc)

    print("\nClassification Report")
    print(classification_report(y_test, pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save model
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nBest Model Saved Successfully")
print("Best Accuracy:", best_accuracy)