from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import numpy
from tqdm import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from pprint import pprint
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ori_df = pd.read_csv("data/Symptom2Disease.csv")
df = ori_df.sample(frac=1.0, random_state=42)  # Set random_state for reproducibility

def get_embeddings(df):
    embedder = OpenAIEmbeddings()

    embed_list = []
    for text, label in tqdm(zip(df.text, df.label), total=len(df)):
        # total = f"{text.lower()} disease={label.lower()}"
        embed = embedder.embed_query(text)
        embed_list.append(embed)
    return embed_list


def get_2d_correlation_matrix(embed_list):
    sz = len(embed_list)
    cor2 = numpy.zeros((sz, sz))

    for i in range(sz):
        for j in range(i+1, sz):
            cor2[i][j] = numpy.dot(embed_list[i], embed_list[j])
            cor2[j][i] = cor2[i][j]
    return cor2


def get_top_n_values_and_indices(data, top_n):
    top_indices = numpy.argsort(-data, axis=1)[:, :top_n]
    top_values = numpy.take_along_axis(data, top_indices, axis=1)
    return top_values, top_indices


top_n = 3

embed_list = get_embeddings(df)
cor2 = get_2d_correlation_matrix(embed_list)
_, top_indices = get_top_n_values_and_indices(cor2, top_n)

labels = df.label.to_list()
texts = df.text.to_list()

cnt,cnt2, cnt3 = 0,0,0
for idx, label in enumerate(labels):
    top_labels = [labels[i] for i in top_indices[idx]]
    # top_texts = [texts[i] for i in top_indices[idx]]
    print(label, top_labels)
    # print(texts[idx])
    # pprint(top_texts)
    # print("-"*100)
    if label in top_labels:
        cnt+=1
    if top_labels.count(label) >= 2:
        cnt2+=1
    if top_labels.count(label) >= 3:
        cnt3+=1

print("at least one match", cnt/len(labels))
print("at least two matches", cnt2/len(labels))
print("all matched", cnt3/len(labels))
