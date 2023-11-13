import os

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain import hub
from langchain.chains.summarize import load_summarize_chain
from google.cloud import aiplatform

project_id = "career-init"
aiplatform.init(project=project_id)


def read_book_pdf(file_path):
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=100,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    token_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    pages2 = pages[10:20]
    # docs = r_splitter.split_documents(pages2)
    docs = token_splitter.split_documents(pages2)

    # google_api_key = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    # chat = ChatVertexAI(
    #     # temperature=0.0,
    #     google_api_key=google_api_key,
    #     model="codechat-bison",
    #     max_output_tokens=2048,
    # )

    from langchain.chains.summarize import load_summarize_chain

    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    map_prompt = """Write a concise summary of the following:
    {text}
    """

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """Write a concise summary of the following text delimited by the 
    triple backtick (```) character:
    Return your responses as bullet points which covers the key points of the text.
    ```{text}```
    bullet points:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    sum_chain = load_summarize_chain(llm=chat,
                                     chain_type="map_reduce",
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                     verbose=True)
    out3 = sum_chain.run(docs)

    # sum_chain = load_summarize_chain(llm=chat,
    reduce_chain = load_summarize_chain(llm=chat,
                                        chain_type="stuff",
                                        combine_prompt=combine_prompt_template,
                                        verbose=True)

    return docs