import os 
import sys


from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator 
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import OpenAIEmbedding

import constants

from dotenv import load_dotenv
import os

os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY

query = sys.argv[1]

loader = DirectoryLoader('.', glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))