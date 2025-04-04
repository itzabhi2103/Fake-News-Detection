import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download NLTK data
nltk.download('stopwords')


# Load datasets
@st.cache_data
def load_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        # List files in ZIP
        file_list = z.namelist()
        
        # Check if our files exist
        if 'True.csv' not in file_list or 'Fake.csv' not in file_list:
            raise FileNotFoundError("Required CSV files not found in ZIP archive")
        
        # Read True.csv
        with z.open('True.csv') as f:
            df_True = pd.read_csv(io.BytesIO(f.read()))
        
        # Read Fake.csv
        with z.open('Fake.csv') as f:
            df_False = pd.read_csv(io.BytesIO(f.read()))

    df_True['label'] = 'True'
    df_False['label'] = 'Fake'

    df = pd.concat([df_True, df_False], ignore_index=True)
    df['Content'] = df['subject'] + ' ' + df['title'] + ' ' + df['text']
    df['label'] = df['label'].map({'Fake': 1, 'True': 0})
    return df


df = load_data()

# Text preprocessing
stop_words = set(stopwords.words('english'))
pt = PorterStemmer()


def preprocess_text(text):
    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize, remove stopwords, and stem
    words = text.split()
    words = [pt.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)


df['Content_processed'] = df['Content'].apply(preprocess_text)

# Vectorization and modeling
X = df['Content_processed'].values
y = df['label'].values

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title('Fake News Detector')
st.write(f"Model Accuracy: {accuracy:.2%}")

input_text = st.text_area('Enter news article:', height=200)


def predict_news(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    proba = model.predict_proba(text_vector)[0]
    return prediction, proba


if st.button('Check News'):
    if input_text.strip() == '':
        st.warning('Please enter some text to analyze.')
    else:
        prediction, proba = predict_news(input_text)

        st.subheader('Result:')
        if prediction == 1:
            st.error('🚨 This news is likely FAKE')
        else:
            st.success('✅ This news is likely REAL')

        st.write('Confidence:')
        st.write(f"- Fake: {proba[1]:.2%}")
        st.write(f"- Real: {proba[0]:.2%}")

        # Add explanation
        threshold = 0.6
        if max(proba) < threshold:
            st.warning('⚠️ The model is uncertain about this prediction (confidence < 60%)')
