import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()

# Retrieve MODEL_NAME environment variable
model = os.getenv("MODEL_NAME")
base_url = os.getenv("LANGFUSE_HOST")

# Initialize Ollama with the model name
llm = Ollama(model=model)

# Create document loader for a PDF in the ./dpocs directory
file_path = (
    "./docs/defra.pdf"
)

loader = PyPDFLoader(file_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
pages = loader.load_and_split(text_splitter)

pages

# Create embeddings
embeddings = (
    OllamaEmbeddings(model='llama3')
)
page_embeddings = embeddings.embed_query(pages)
page_embeddings

# Create vector store
# vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings)

# Create retrieval chain
# qachain = create_retrieval_chain(llm, retriever=vectorstore.as_retriever())

# question = "what is the ETC of each vessel?"
# response = qachain({"query": question})

# print(response)
