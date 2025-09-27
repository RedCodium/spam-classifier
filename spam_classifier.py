import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load dataset
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

print("Dataset loaded successfully ✅")
print(df.head())

# 2. Split into features (X) and labels (y)
X = df['message']
y = df['label']

# 3. Split into train & test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 4. Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Text vectorization complete ✅")
print("Shape of training data:", X_train_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 5. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model training complete ✅")

# 6. Make predictions on test data
y_pred = model.predict(X_test_tfidf)

# 7. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

# Save model
joblib.dump(model, "spam_model.joblib")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("Model and vectorizer saved successfully ✅")

