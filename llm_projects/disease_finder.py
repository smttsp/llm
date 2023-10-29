import numpy
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm import tqdm


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
        for j in range(i + 1, sz):
            cor2[i][j] = numpy.dot(embed_list[i], embed_list[j])
            cor2[j][i] = cor2[i][j]
    return cor2


def get_top_n_values_and_indices(data, top_n):
    top_indices = numpy.argsort(-data, axis=1)[:, :top_n]
    top_values = numpy.take_along_axis(data, top_indices, axis=1)
    return top_values, top_indices
