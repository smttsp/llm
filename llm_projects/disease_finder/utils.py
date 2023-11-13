"""This one creates a vectorstore from the training data by preprocessing the
input text using the two common preprocessing steps:

# https://chat.openai.com/share/4ada2e1b-7df8-4039-aab3-7ffef441ea3e

1. Lowercase
2. Remove stopwords
3. Lemmatize

"""
import nltk
from langchain.embeddings.openai import OpenAIEmbeddings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download the NLTK stop words data
nltk.download("stopwords")
nltk.download("punkt")

# Get the list of English stop words
STOP_WORDS = set(stopwords.words("english"))

# Download the WordNet lemmatizer data
nltk.download("wordnet")

# Initialize the lemmatizer
LEMMATIZER = WordNetLemmatizer()


def lemmatize(text, lemmatizer=None):
    if lemmatizer is None:
        lemmatizer = LEMMATIZER

    # Tokenize the sentence into words
    words = nltk.word_tokenize(text)

    # Lemmatize each word in the sentence
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Join the lemmatized words back into a sentence
    lemmatized_sentence = " ".join(lemmatized_words)

    # Print the lemmatized sentence
    # print(lemmatized_sentence)
    return lemmatized_sentence


def remove_stopwords(text, stop_words=None):
    if stop_words is None:
        stop_words = STOP_WORDS

    # Tokenize the sentence into words
    words = word_tokenize(text)

    # Remove stop words from the tokenized words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a sentence
    filtered_sentence = " ".join(filtered_words)

    # Print the filtered sentence
    # print(filtered_sentence)
    return filtered_sentence


class OpenAIEmbeddingsStopWord(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def embed_query(self, text):
        preprocessed_text = remove_stopwords(text)
        # Call the original embed_query method with the preprocessed text
        return super().embed_query(preprocessed_text)

    def embed_documents(self, texts, **kwargs):
        preprocessed_texts = [remove_stopwords(text) for text in texts]
        # Call the original embed_query method with the preprocessed text
        return super().embed_documents(preprocessed_texts, **kwargs)


class OpenAIEmbeddingsLemmatize(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def embed_query(self, text):
        preprocessed_text = lemmatize(text)
        # Call the original embed_query method with the preprocessed text
        return super().embed_query(preprocessed_text)

    def embed_documents(self, texts, **kwargs):
        preprocessed_texts = [lemmatize(text) for text in texts]
        # Call the original embed_query method with the preprocessed text
        return super().embed_documents(preprocessed_texts, **kwargs)


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
