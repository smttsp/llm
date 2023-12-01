import os

from dotenv import find_dotenv, load_dotenv

from llm_projects.healthcare.read_mimic import read_csvs


load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    folder = "/users/samet/desktop/mimic-iv-2.2/note/"
    read_csvs(folder)
