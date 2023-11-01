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

    index_id = 0
    #
    # embedding_vector2 = vector_db2.index.reconstruct_n(index_id, 1)[0]
    embeds = vector_db.index.reconstruct_n(index_id, 1)[0]
    # embeds = vector_db._collection.get(include=['embeddings', "metadatas"])
    print(embeds[:3])

    # vector_db2 = FAISS.from_documents(
    #     documents=docs, embedding=OpenAIEmbeddingsLower()
    # )
    # embeds2 = vector_db2.index.reconstruct_n(index_id, 1)[0]
    # # embeds2 = vector_db2._collection.get(include=['embeddings', "metadatas"])
    # print(embeds2[:3])

    # vectordb._persist_directory = persist_directory
    return vector_db


def disease_finder_v2():
    vector_db = get_vectorstore("data/test.csv")
    df = pandas.read_csv("data/test.csv")

    # embeds = vector_db._collection.get(include=['embeddings', "metadatas"])
    oa_embedder = OpenAIEmbeddings()
    # oa_embedder = OpenAIEmbeddingsLower()

    # cur_text = df.text[idx]
    # txt_embed = oa_embedder.embed_query(cur_text)
    # embedding_vector = vector_db.index.reconstruct_n(idx, 1)[0]
    mx = -1

    idx = 0
    cur_text = df.text[idx]
    print(df.label[idx], "---", cur_text)
    txt_embed = oa_embedder.embed_query(cur_text)
    index_to_docstore_id = vector_db.index_to_docstore_id
    for idx in range(len(df)):
        embedding_vector = vector_db.index.reconstruct_n(idx, 1)[0]
        metadata = vector_db.docstore._dict[index_to_docstore_id[idx]]
        cur_cor = numpy.dot(embedding_vector, txt_embed)
        if cur_cor > mx:
            mx = cur_cor
            print(cur_cor, df.label[idx], "---", cur_text)
            print("*" * 30)
            print()


    cnt, cnt2, cnt3 = 0, 0, 0
    top_n = 10
    for text, label in zip(df.text, df.label):

        # res = vector_db.get_relevant_documents(text, top_n)
        top_n_results = vector_db.max_marginal_relevance_search(text.lower(), fetch_k=top_n, k=3)
        pred_labels = [doc.metadata.get("label") for doc in top_n_results]
        print(f"{label=}, {pred_labels=}")

        # print("*" * 30)
        cnt += int(label in pred_labels)
        cnt2 += int(pred_labels.count(label) >= 2)
        cnt3 += int(pred_labels.count(label) >= 3)

    print("at least one match", cnt / len(df))
    print("at least two matches", cnt2 / len(df))
    print("all matched", cnt3 / len(df))
