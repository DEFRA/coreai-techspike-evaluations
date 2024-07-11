import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain

# import psycopg2
from langchain_core.documents import Document
# from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

# Load environment variables from .env file
load_dotenv()

# Retrieve MODEL_NAME environment variable
model = os.getenv("MODEL_NAME")
base_url = os.getenv("LANGFUSE_HOST")

# Initialize Ollama with the model name
llm = Ollama(model=model)

# Create document loader for a PDF in the ./docs directory
file_path = "./docs/defra.pdf"

# Load and split the document
loader = PyPDFLoader(file_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

# TEMP for local testing - added to reduce time for saving embeddings to PGVector
# docs = docs[0:2]

# Create embeddings
embeddings = OllamaEmbeddings(model='llama3')

doc_vectors = embeddings.embed_documents([t.page_content for t in docs[:5]]) # update :1 to :5 to return 5 documents

# Vectorstore
connection_string = f"postgresql+psycopg://{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

vectorstore = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    connection=connection_string,
    collection_name=os.getenv('PGVECTOR_COLLECTION_NAME'),
    use_jsonb=True,
    async_mode=False,
)

# Query
query = "What is the budget?"

similar = vectorstore.similarity_search_with_score(query, k=2) # k=2 to return 2 similar documents

for doc in similar:
  print(doc)
  print(f"Page: {doc[0].metadata['page']}, Similarity: {doc[1]}")