import zipfile
import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, request, render_template
import re

app = Flask(__name__)

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to load emails and assign labels
def load_emails(directory):
    emails = []
    for file_path in glob.glob(os.path.join(directory, "*")):
        label = 'spam' if 'spm' in os.path.basename(file_path).lower() else 'ham'
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            content = clean_text(content)  # Clean the content
            emails.append((content, label))
            print(f"Loaded: {file_path}, Label: {label}")
    return emails

# Load models
def load_models():
    zip_path = './train_test_mails.zip'
    extract_dir = './extracted_data'

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Paths to the extracted train-mails and test-mails directories
    train_dir = os.path.join(extract_dir, 'train-mails')
    test_dir = os.path.join(extract_dir, 'test-mails')

    # Load training and test emails
    train_emails = load_emails(train_dir)
    test_emails = load_emails(test_dir)

    # Convert to DataFrame
    train_df = pd.DataFrame(train_emails, columns=['text', 'label'])
    test_df = pd.DataFrame(test_emails, columns=['text', 'label'])

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer

# Load models at startup
model, vectorizer = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email_content']
    email_content_cleaned = clean_text(email_content)
    email_vectorized = vectorizer.transform([email_content_cleaned])
    
    prediction = model.predict(email_vectorized)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
