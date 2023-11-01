"""This one creates a vectorstore from the training data, and then uses it to
find the top N most similar documents to each document in the test data.

Then, finds the closest k documents to each document in the test data.

It has two strategies for determining the predicted label:
1. Majority vote
2. pick the closest M documents, and then pick the label that appears the most
"""

import pandas

from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader

from .constants import VECTOR_DB_DIRECTORY
from langchain.embeddings.openai import OpenAIEmbeddings


def get_chroma_vectorstore(file_path, persist_directory=VECTOR_DB_DIRECTORY):
    embedding = OpenAIEmbeddings()
    loader = CSVLoader(
        file_path=file_path, metadata_columns=["label", "text"]
    )
    docs = loader.load()

    vectordb = Chroma.from_documents(
        documents=docs, embedding=embedding #, persist_directory=persist_directory
    )
    vectordb._persist_directory = persist_directory
    return vectordb


def disease_finder_v2():
    vector_db = get_chroma_vectorstore("data/train.csv")
    df = pandas.read_csv("data/test.csv")

    cnt, cnt2, cnt3 = 0, 0, 0
    top_n = 50
    for text, label in zip(df.text, df.label):
        top_n_results = vector_db.similarity_search(text, top_n)

        pred_labels = [doc.metadata.get("label") for doc in top_n_results]
        print(f"{label=}, {pred_labels=}")

        break
        # print("*" * 30)
        cnt += int(label in pred_labels)
        cnt2 += int(pred_labels.count(label) >= 2)
        cnt3 += int(pred_labels.count(label) >= 3)

    print("at least one match", cnt / len(df))
    print("at least two matches", cnt2 / len(df))
    print("all matched", cnt3 / len(df))
