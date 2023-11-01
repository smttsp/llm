"""This one creates a vectorstore from the training data, and then uses it to
find the top N most similar documents to each document in the test data.

Then, finds the closest k documents to each document in the test data.

It has two strategies for determining the predicted label:
1. Majority vote
2. pick the closest M documents, and then pick the label that appears the most
"""

import pandas
from pprint import pprint
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader

from .constants import VECTOR_DB_DIRECTORY
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy


def get_chroma_vectorstore(file_path, persist_directory=VECTOR_DB_DIRECTORY):
    oa_embedder = OpenAIEmbeddings()
    loader = CSVLoader(
        file_path=file_path, metadata_columns=["label", "text"]
    )
    docs = loader.load()

    vectordb = Chroma.from_documents(
        documents=docs, embedding=oa_embedder #, persist_directory=persist_directory
    )
    # vectordb = Milvus.from_documents(
    #     docs,
    #     embedding=embedding,
    #     # connection_args={"uri": "Use your uri:)", "token": "Use your token:)"},
    # )

    # vectordb._persist_directory = persist_directory
    return vectordb


def disease_finder_v2():
    vector_db = get_chroma_vectorstore("data/train.csv")
    df = pandas.read_csv("data/test.csv")

    embeds = vector_db._collection.get(include=['embeddings', "metadatas"])

    oa_embedder = OpenAIEmbeddings()
    txt_embed = oa_embedder.embed_query(df.text[1])

    mx = -1
    for embed, meta in zip(embeds["embeddings"], embeds["metadatas"]):
        cur_cor = numpy.dot(embed, txt_embed)
        if cur_cor > mx:
            mx = cur_cor
            print(cur_cor, meta["label"], "---", meta["text"])
            e1 = oa_embedder.embed_query(meta["text"].lower())
            e2 = oa_embedder.embed_query(df.text[1].lower())
            print(numpy.dot(e1, e2))
            print("*" * 30)
            print()

    cnt, cnt2, cnt3 = 0, 0, 0
    top_n = 250
    for text, label in zip(df.text, df.label):

        res = vector_db.get_relevant_documents(text, top_n)
        top_n_results = vector_db.max_marginal_relevance_search(text, fetch_k=top_n, k=50)
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
