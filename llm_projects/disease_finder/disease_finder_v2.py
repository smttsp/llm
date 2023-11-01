"""This one creates a vectorstore from the training data, and then uses it to
find the top N most similar documents to each document in the test data.

Then, finds the closest k documents to each document in the test data.

It has two strategies for determining the predicted label:
1. Majority vote
2. pick the closest M documents, and then pick the label that appears the most
"""

import pandas
from langchain.vectorstores import FAISS

from langchain.document_loaders.csv_loader import CSVLoader

from .constants import VECTOR_DB_DIRECTORY
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy


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

    for text, label in zip(df.text, df.label):
        embedding_vector = OpenAIEmbeddings().embed_query(text)
        top_n_results = vector_db.similarity_search_by_vector(embedding_vector, k=5)

        # top_n_results = vector_db.max_marginal_relevance_search(
        #    text, fetch_k=top_n, k=k
        # )
        pred_labels = [doc.metadata.get("label") for doc in top_n_results]
        print(f"{label=}, {pred_labels=}")

        for i in range(k):
            cnts[i] += int(pred_labels.count(label) > i)

    for i in range(k):
        print(f"at least {i+1} match", cnts[i] / len(df))

    return cnts / len(df)
