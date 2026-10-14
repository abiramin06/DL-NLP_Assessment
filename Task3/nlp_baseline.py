import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import gensim.downloader as api
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

positive_texts = [
    "I love this product",
    "Excellent customer service",
    "Very happy with the purchase",
    "The quality is amazing",
    "Fast delivery and great packaging",
    "This product works perfectly",
    "Wonderful experience",
    "The support team was helpful",
    "Highly recommend this product",
    "Great value for money",
    "I am satisfied with the service",
    "Fantastic quality and performance",
    "The delivery was quick",
    "Easy to use and reliable",
    "Absolutely wonderful experience",
    "The product exceeded expectations",
    "Friendly and professional support",
    "Very useful and convenient",
    "Good quality for the price",
    "I would buy this again"
]

negative_texts = [
    "I hate this product",
    "Terrible customer service",
    "Very disappointed with the purchase",
    "The quality is awful",
    "Slow delivery and poor packaging",
    "This product does not work",
    "Horrible experience",
    "The support team was unhelpful",
    "I do not recommend this product",
    "Poor value for money",
    "I am unhappy with the service",
    "Terrible quality and performance",
    "The delivery was late",
    "Difficult to use and unreliable",
    "Absolutely disappointing experience",
    "The product failed my expectations",
    "Rude and unprofessional support",
    "Very useless and inconvenient",
    "Bad quality for the price",
    "I would never buy this again"
]

positive_samples = positive_texts * 10
negative_samples = negative_texts * 10

texts = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

df = pd.DataFrame({
    "text": texts,
    "label": labels})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

print("Dataset size:", len(df))
print("\nClass distribution:",df["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(df["text"],df["label"],test_size=0.2,random_state=42,stratify=df["label"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("\nTF-IDF Baseline")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nConfusion Matrix:",confusion_matrix(y_test, predictions))
print("\nClassification Report:",classification_report(y_test, predictions, zero_division=0))

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]
feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

positive_indices = np.argsort(coefficients)[-10:][::-1]
negative_indices = np.argsort(coefficients)[:10]

print("\nTop 10 positive words:",feature_names[positive_indices])
print("\nTop 10 negative words:",feature_names[negative_indices])

glove = api.load("glove-wiki-gigaword-50")

for word in ["good", "bad"]:
    print(f"\n5 nearest words to '{word}':")
    for similar_word, score in glove.most_similar(word, topn=5):
        print(similar_word, round(score, 3))

print("\nWord analogy: king - man + woman")
analogy = glove.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
print("Result:", analogy[0][0])

def sentence_vector(sentence):
    words = sentence.lower().split()

    vectors = [
        glove[word]
        for word in words
        if word in glove
    ]

    if not vectors:
        return np.zeros(50, dtype=np.float32)

    return np.mean(vectors, axis=0)

X_train_vectors = np.array([
    sentence_vector(text) for text in X_train
])

X_test_vectors = np.array([
    sentence_vector(text) for text in X_test
])

print("\nTraining vector shape:", X_train_vectors.shape)
print("Testing vector shape:", X_test_vectors.shape)

embedding_model = LogisticRegression(max_iter=1000)
embedding_model.fit(X_train_vectors, y_train)

embedding_predictions = embedding_model.predict(X_test_vectors)
embedding_accuracy = accuracy_score(
    y_test, embedding_predictions
)
embedding_f1 = f1_score(
    y_test, embedding_predictions, average="macro"
)
print("\nModel Comparison")
print("-" * 48)
print(f"{'Model':<22}{'Accuracy':<13}{'Macro F1':<13}")
print("-" * 48)
print(f"{'TF-IDF':<22}{accuracy:.4f}       {macro_f1:.4f}")
print(
    f"{'Average GloVe':<22}"
    f"{embedding_accuracy:.4f}       {embedding_f1:.4f}")