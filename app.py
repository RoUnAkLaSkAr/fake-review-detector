```python
import streamlit as st
import joblib
import nltk
import re
import string
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Fake Product Review Detector",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------
# DOWNLOAD NLTK RESOURCES
# ----------------------------

@st.cache_resource
def load_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

load_nltk()

# ----------------------------
# LOAD MODEL
# ----------------------------

model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------------------
# NLP SETUP
# ----------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# TEXT CLEANING
# ----------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
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

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("ℹ️ About")

st.sidebar.info(
    """
    Fake Product Review Detection System

    Model:
    Logistic Regression

    Features:
    • TF-IDF
    • NLP Processing
    • Sentiment Analysis
    • Review Statistics

    Developed using:
    Streamlit + Scikit-Learn
    """
)

# ----------------------------
# TITLE
# ----------------------------

st.title("🛒 Fake Product Review Detection System")

st.markdown(
    """
    Detect whether an e-commerce review is
    **Genuine** or **Fake**
    using Machine Learning and NLP.
    """
)

# ----------------------------
# LAYOUT
# ----------------------------

col1, col2 = st.columns([2, 1])

with col1:

    review = st.text_area(
        "✍️ Enter Product Review",
        height=200
    )

with col2:

    st.info(
        """
        Example Reviews

        ✅ Battery lasts 8 hours.
        Packaging was good.

        ⚠ Amazing product!!!
        Best purchase ever!!!
        Buy now!!!
        """
    )

# ----------------------------
# PREDICT BUTTON
# ----------------------------

if st.button("🔍 Analyze Review"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        cleaned_review = clean_text(review)

        vector = vectorizer.transform(
            [cleaned_review]
        )

        prediction = model.predict(
            vector
        )[0]

        # Confidence score

        try:

            confidence = (
                model.predict_proba(vector).max()
                * 100
            )

        except:

            confidence = 0

        # ----------------------------
        # RESULT
        # ----------------------------

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠ Fake Review Detected"
            )

            result_text = "Fake Review"

        else:

            st.success(
                "✅ Genuine Review"
            )

            result_text = "Genuine Review"

        st.write(
            f"Confidence Score: {confidence:.2f}%"
        )

        st.progress(
            min(confidence / 100, 1.0)
        )

        # ----------------------------
        # REVIEW METRICS
        # ----------------------------

        st.subheader("📊 Review Statistics")

        word_count = len(review.split())

        char_count = len(review)

        exclamation_count = review.count("!")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Words",
            word_count
        )

        m2.metric(
            "Characters",
            char_count
        )

        m3.metric(
            "Exclamation Marks",
            exclamation_count
        )

        # ----------------------------
        # SENTIMENT ANALYSIS
        # ----------------------------

        st.subheader("😊 Sentiment Analysis")

        sentiment = TextBlob(
            review
        ).sentiment.polarity

        if sentiment > 0:

            st.success(
                "Positive Review"
            )

        elif sentiment < 0:

            st.error(
                "Negative Review"
            )

        else:

            st.warning(
                "Neutral Review"
            )

        # ----------------------------
        # FEATURE CHART
        # ----------------------------

        st.subheader("📈 Review Feature Chart")

        chart_df = pd.DataFrame({

            "Feature": [

                "Words",
                "Characters",
                "Exclamation Marks"

            ],

            "Value": [

                word_count,
                char_count,
                exclamation_count

            ]

        })

        st.bar_chart(
            chart_df.set_index(
                "Feature"
            )
        )

        # ----------------------------
        # WORD CLOUD
        # ----------------------------

        st.subheader("☁️ Word Cloud")

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(review)

        fig, ax = plt.subplots()

        ax.imshow(
            wc,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(fig)

        # ----------------------------
        # HISTORY
        # ----------------------------

        if "history" not in st.session_state:

            st.session_state.history = []

        st.session_state.history.append({

            "Review":
            review[:60],

            "Prediction":
            result_text

        })

# ----------------------------
# HISTORY TABLE
# ----------------------------

if "history" in st.session_state:

    st.subheader("🕒 Prediction History")

    st.dataframe(
        pd.DataFrame(
            st.session_state.history
        )
    )
```
