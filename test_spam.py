import joblib

# Load model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Test custom messages
messages = ["You won $5000! Click here!", "Let's meet for lunch tomorrow."]
msgs_tfidf = vectorizer.transform(messages)
predictions = model.predict(msgs_tfidf)

for msg, label in zip(messages, predictions):
    print(f"Message: {msg}\nPrediction: {label}\n")
