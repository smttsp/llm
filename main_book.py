import os

from dotenv import find_dotenv, load_dotenv

from llm_projects.book_summarizer import read_book_pdf


load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    book_path = "data/books/lotr.pdf"
    read_book_pdf(book_path)
