import joblib

# Load saved model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

print("=== Spam Classifier Interactive ===")
print("Type your message and press Enter. Type 'exit' to quit.\n")

while True:
    msg = input("Your message: ")
    if msg.lower() == "exit":
        print("Goodbye! 👋")
        break

    # Transform the message
    msg_tfidf = vectorizer.transform([msg])
    prediction = model.predict(msg_tfidf)[0]

    print(f"Prediction: {prediction}\n")
