# Fake E-Commerce Product Review Detection System

## Project Overview

This project detects whether an e-commerce product review is genuine or fake using Machine Learning and Natural Language Processing (NLP).

## Features

- Text preprocessing
- TF-IDF Vectorization
- Sentiment Analysis
- Feature Engineering
- Machine Learning Classification
- Streamlit Web Application

## Dataset

Dataset Source:
https://www.kaggle.com/datasets/muqaddasejaz/fake-reviews-dataset

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- NLTK
- TextBlob
- Streamlit

## Project Structure

```text
FakeReviewDetection/

├── app.py
├── requirements.txt
├── fake_review_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
└── fake_reviews_dataset.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Model

The model is trained using:

- TF-IDF Vectorizer
- Logistic Regression

## Example

Input:

Amazing product. Buy now!!!

Output:

Fake Review

## Author

Rounak Laskar
