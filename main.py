import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from llm_projects.disease_finder import (
    get_2d_correlation_matrix,
    get_embeddings,
    get_top_n_values_and_indices,
)


load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ori_df = pd.read_csv("data/Symptom2Disease.csv")
df = ori_df.sample(
    frac=1.0, random_state=42
)  # Set random_state for reproducibility


top_n = 3

embed_list = get_embeddings(df)
cor2 = get_2d_correlation_matrix(embed_list)
_, top_indices = get_top_n_values_and_indices(cor2, top_n)

labels = df.label.to_list()
texts = df.text.to_list()

cnt, cnt2, cnt3 = 0, 0, 0
for idx, label in enumerate(labels):
    top_labels = [labels[i] for i in top_indices[idx]]
    # top_texts = [texts[i] for i in top_indices[idx]]
    print(label, top_labels)
    # print(texts[idx])
    # pprint(top_texts)
    # print("-"*100)
    if label in top_labels:
        cnt += 1
    if top_labels.count(label) >= 2:
        cnt2 += 1
    if top_labels.count(label) >= 3:
        cnt3 += 1

print("at least one match", cnt / len(labels))
print("at least two matches", cnt2 / len(labels))
print("all matched", cnt3 / len(labels))
