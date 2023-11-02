"""Together with what has been done in v2, this version adds the integration of
ChatGPT for which uses the similar docs that are found in the first step
to generate the answer for the questions, i.e. figuring out the disease.
"""

import pandas
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy

from .disease_finder_v2 import get_vectorstore
from langchain.llms import OpenAI
from langchain.retrievers.merger_retriever import MergerRetriever


def disease_finder_v3():
    vector_db = get_vectorstore("data/train.csv")
    df = pandas.read_csv("data/test.csv")

    k = 5
    cnts = numpy.zeros(k)

    for text, label in zip(df.text, df.label):
        retriever_all = vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5, "include_metadata": True}
        )

        top_n_results = retriever_all.get_relevant_documents(text)

        pred_labels = [doc.metadata.get("label") for doc in top_n_results]
        print(f"{label=}, {pred_labels=}")

        for i in range(k):
            cnts[i] += int(pred_labels.count(label) > i)

    for i in range(k):
        print(f"at least {i + 1} match {cnts[i] / len(df):.3f}")

    return cnts / len(df)
