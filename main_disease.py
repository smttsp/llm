import os

from dotenv import find_dotenv, load_dotenv

from llm_projects.disease_finder import (
    disease_finder_v3, disease_finder_v2
)

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    disease_finder_v3()
