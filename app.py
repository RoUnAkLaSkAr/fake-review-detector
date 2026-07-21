import streamlit as st
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load(
    'fake_review_model.pkl'
)

vectorizer = joblib.load(
    'tfidf_vectorizer.pkl'
)

stop_words = set(
    stopwords.words('english')
)

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'\d+',
        '',
        text
    )

    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    words = text.split()

    words = [

        lemmatizer.lemmatize(word)

        for word in words

        if word not in stop_words

    ]

    return " ".join(words)

st.title(
    "Fake Product Review Detection"
)

review = st.text_area(
    "Enter Product Review"
)

if st.button("Predict"):

    review = clean_text(
        review
    )

    vector = vectorizer.transform(
        [review]
    )

    prediction = model.predict(
        vector
    )[0]

    if prediction == 1:

        st.error(
            "⚠ Fake Review Detected"
        )

    else:

        st.success(
            "✓ Genuine Review"
        )