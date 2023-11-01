"""This one creates a vectorstore from the training data by preprocessing the
input text using the two common preprocessing steps:

# https://chat.openai.com/share/4ada2e1b-7df8-4039-aab3-7ffef441ea3e

1. Lowercase
2. Remove stopwords
3. Lemmatize

"""

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


# # Example usage
# oa_embedder = CustomOpenAIEmbeddings()
# txt_embed = oa_embedder.embed_query(df.text[1])
