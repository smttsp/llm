"""This one creates a vectorstore from the training data, and then uses it to
find the top N most similar documents to each document in the test data.

Then, finds the closest k documents to each document in the test data.

It has two strategies for determining the predicted label:
1. Majority vote
2. pick the closest M documents, and then pick the label that appears the most
"""

import pandas
from pprint import pprint
from langchain.vectorstores import Chroma, FAISS

from langchain.document_loaders.csv_loader import CSVLoader

from .constants import VECTOR_DB_DIRECTORY
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy


class OpenAIEmbeddingsLower(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess_text(self, text):
        # Add your preprocessing steps here, for example, converting text to lowercase
        return text.lower()

    def embed_query(self, text):
        preprocessed_text = self.preprocess_text(text)
        # Call the original embed_query method with the preprocessed text
        return super().embed_query(preprocessed_text)

    def embed_documents(self, texts, **kwargs):
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        # Call the original embed_query method with the preprocessed text
        return super().embed_documents(preprocessed_texts, **kwargs)


def get_vectorstore(file_path, persist_directory=VECTOR_DB_DIRECTORY):
    loader = CSVLoader(
        file_path=file_path, source_column="text", metadata_columns=["label"]
    )
    docs = loader.load()

    vector_db = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(),
    )

    # vectordb._persist_directory = persist_directory
    return vector_db


def disease_finder_v2():
    vector_db = get_vectorstore("data/train.csv")
    df = pandas.read_csv("data/test.csv")

    k = 5
    cnts = numpy.zeros(k)

    top_n = 10
    for text, label in zip(df.text, df.label):
        top_n_results = vector_db.max_marginal_relevance_search(text, fetch_k=top_n, k=k)
        pred_labels = [doc.metadata.get("label") for doc in top_n_results]
        print(f"{label=}, {pred_labels=}")

        for i in range(k):
            cnts[i] += int(pred_labels.count(label) > i)

    for i in range(k):
        print(f"at least {i+1} match", cnts[i] / len(df))

    return cnts / len(df)