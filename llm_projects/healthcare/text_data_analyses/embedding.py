import numpy
from langchain.embeddings.openai import OpenAIEmbeddings


def get_embeddings(df):
    df = df.drop_duplicates(subset=['subject_id'])

    embedder = OpenAIEmbeddings()

    embeds = []
    for idx, row in df.iterrows():
        # print(idx, row)
        embed = embedder.embed_query(row["text"])
        embeds.append(embed)

        if idx == 100:
            break

    sz = len(embeds)
    cor2 = numpy.zeros((sz, sz))

    for i in range(len(embeds)):
        for j in range(i + 1, len(embeds)):
            cor2[i][j] = numpy.dot(embeds[i], embeds[j])
            cor2[j][i] = cor2[i][j]

    x = cor2.max(axis=1)
    print(x)
    return None