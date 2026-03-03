import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (only first time)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

# Remove null or empty cleaned messages
df.dropna(subset=['cleaned_message'], inplace=True)
df = df[df['cleaned_message'].str.strip() != ""]

# Features and Labels
X = df['cleaned_message']
y = df['label']

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Trained Successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model/scam_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model and Vectorizer saved successfully!")