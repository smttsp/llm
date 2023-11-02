"""Together with what has been done in v2, this version adds the integration of
ChatGPT for which uses the similar docs that are found in the first step
to generate the answer for the questions, i.e. figuring out the disease.
"""

import pandas
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy

from .disease_finder_v2 import get_vectorstore


def disease_finder_v3():
    vector_db = get_vectorstore("data/train.csv")
    df = pandas.read_csv("data/test.csv")

    k = 5
    cnt = 0

    oe_embedder = OpenAIEmbeddings()
    for text, label in zip(df.text, df.label):
        embedding_vector = oe_embedder.embed_query(text)
        top_n_results = vector_db.similarity_search_by_vector(embedding_vector, k=k)

        # Ask chatgpt for the answer

    print(cnt / len(df), "of them accurately predicted")
    return cnt / len(df)
