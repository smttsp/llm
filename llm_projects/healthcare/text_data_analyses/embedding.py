import numpy
from langchain.embeddings.openai import OpenAIEmbeddings
from pprint import pprint


def compare_single_embedding(embed_list, gt_docs, cur_doc):
    embedding = OpenAIEmbeddings()
    cur_embed = embedding.embed_query(cur_doc.page_content)

    cors = numpy.zeros(len(gt_docs))
    for i, doc in enumerate(gt_docs):
        if doc.metadata["subject_id"] == cur_doc.metadata["subject_id"]:
            continue

        cors[i] = numpy.dot(embed_list[i], cur_embed)

    top_n = 3
    top_indices = numpy.argsort(cors)[-top_n:][::-1]
    top_values = numpy.take_along_axis(cors, top_indices, axis=0)

    print(top_values, top_indices)
    pprint(cur_doc.page_content)

    for t in top_indices:
        pprint(gt_docs[t].page_content)

    return sum(top_values)


def find_matching_embeddings(embed_list, gt_docs, new_docs):
    total = 0
    for doc in new_docs:
        total += compare_single_embedding(embed_list, gt_docs, doc)

    print(total/3/len(new_docs))

    return None


def get_embeddings_from_docs(docs):
    embed_list = numpy.load("embed_list.npy")
    # embedder = OpenAIEmbeddings()
    #
    # embed_list = embedder.embed_documents([d.page_content for d in docs])
    # for text in docs:
    #     embed =
    #     embed_list.append(embed)
    return embed_list


def embedder_test(docs):
    gt_docs = docs[:5000]
    new_docs = docs[100000:100100]
    embed_list = get_embeddings_from_docs(gt_docs)
    find_matching_embeddings(embed_list, gt_docs, new_docs)
    return None


def get_embeddings(df):
    df = df.drop_duplicates(subset=["subject_id"])

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
