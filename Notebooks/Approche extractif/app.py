import joblib
import os
import sys
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

class TextRankSummarizer(BaseEstimator, TransformerMixin):

    def __init__(self, top_k=3):

        self.top_k = top_k

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        summaries = []

        for doc in X:

            sim_matrix = doc["similarity_matrix"]

            graph = nx.from_numpy_array(sim_matrix)

            scores = nx.pagerank(
                graph,
                alpha=0.85
            )

            ranked_sentences = sorted(
                (
                    (scores[i], sentence, i)
                    for i, sentence in enumerate(doc["sentences"])
                ),
                reverse=True
            )

            top_sentences = sorted(
                ranked_sentences[:self.top_k],
                key=lambda x: x[2]
            )

            summary = " ".join(
                sentence
                for _, sentence, _
                in top_sentences
            )

            summaries.append(summary)

        return summaries


class SimilarityMatrix(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        for doc in X:

            sim_matrix = cosine_similarity(
                doc["sentence_vectors"]
            )

            np.fill_diagonal(sim_matrix, 0)

            doc["similarity_matrix"] = sim_matrix

        return X



class SentenceVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer_type="tfidf"):

        self.vectorizer_type = vectorizer_type

        if vectorizer_type == "bow":
            self.vectorizer = CountVectorizer()

        elif vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer()

        else:
            raise ValueError(
                "vectorizer_type must be 'bow' or 'tfidf'"
            )

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        for doc in X:

            sentence_vectors = self.vectorizer.fit_transform(
                doc["cleaned_sentences"]
            )

            doc["sentence_vectors"] = sentence_vectors

        return X

class Tokenizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        for doc in X:

            cleaned_sentences = []

            for sentence in doc["sentences"]:

                tokens = word_tokenize(sentence.lower())

                tokens = [
                    token
                    for token in tokens
                    if token.isalnum()
                    and token not in STOPWORDS
                ]

                cleaned_sentences.append(
                    " ".join(tokens)
                )

            doc["cleaned_sentences"] = cleaned_sentences

        return X


class SentenceExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        documents = []

        for text in X:

            sentences = sent_tokenize(text)

            documents.append({
                "original_text": text,
                "sentences": sentences
            })

        return documents

import pickle
import os
import sys

PIPELINE_PATH = r"Notebooks\Approche extractif\pipelines\tfidf_summarizer.pkl"


def load_pipeline():
    try:
        with open(PIPELINE_PATH, "rb") as f:
            pipeline = joblib.load(f)

        print(f"[OK] Pipeline loaded: {type(pipeline).__name__}")
        return pipeline

    except FileNotFoundError:
        print(f"[ERROR] File not found: {PIPELINE_PATH}")
        sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {e}")
        sys.exit(1)


def summarize(pipeline, text):
    try:
        pipeline.fit([text])   # force initialization
        result = pipeline.transform([text])
        return result[0] if isinstance(result, (list, tuple)) else result

    except Exception as e:
        return f"[ERROR] {e}"


def read_pasted_text():
    print("\nPaste your text below.")
    print("When finished, type END on a new line.\n")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    return "\n".join(lines)


def main():
    pipeline = load_pipeline()

    print("\n=== Extractive Text Summarizer ===")
    print("Commands:")
    print("  exit  -> quit")
    print("  file  -> summarize a text file")
    print("  paste -> paste multi-line text")
    print()

    while True:
        choice = input("> ").strip().lower()

        if choice == "exit":
            print("Goodbye.")
            break

        elif choice == "file":
            path = input("File path: ").strip()

            if not os.path.exists(path):
                print("[ERROR] File not found.\n")
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                summary = summarize(pipeline, text)

                print("\n--- SUMMARY ---")
                print(summary)
                print("----------------\n")

            except Exception as e:
                print(f"[ERROR] {e}\n")

        elif choice == "paste":
            text = read_pasted_text()

            summary = summarize(pipeline, text)

            print("\n--- SUMMARY ---")
            print(summary)
            print("----------------\n")

        else:
            if not choice:
                continue

            summary = summarize(pipeline, choice)

            print("\n--- SUMMARY ---")
            print(summary)
            print("----------------\n")


if __name__ == "__main__":
    main()