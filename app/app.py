import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector

from langchain_core.prompts.prompt import PromptTemplate

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Retrieve MODEL_NAME environment variable
model = os.getenv("MODEL_NAME")
base_url = os.getenv("LANGFUSE_HOST")


# Initialize Ollama with the model name
llm = Ollama(model=model)


# # DOCUMENT LOADER --------------------------------------------------------------------------------------

# # # Create document loader for a PDF in the ./docs directory
file_path = "./docs/defra.pdf"

# Load and split the document
loader = PyPDFLoader(file_path)
pages = loader.load()



# TEXT SPLITTER -----------------------------------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

docs = docs[0:2] # TEMP for local testing - added to reduce time for saving embeddings to PGVector



# CREATE AND STORE EMBEDDINGS ----------------------------------------------------------------------------

# Create embeddings
embeddings = OllamaEmbeddings(model='llama3')

doc_vectors = embeddings.embed_documents([t.page_content for t in docs[:5]]) # update :1 to :5 - defines the number of documents to embed

# Populatre the PGVector vectorstore
connection_string = f"postgresql+psycopg://{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
collection_name = os.getenv('PGVECTOR_COLLECTION_NAME')

vectorstore = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    connection=connection_string,
    collection_name=collection_name,
    use_jsonb=True,
    async_mode=False,
)

retriever = vectorstore.as_retriever()


# FIND SIMILAR VECTORS BASED ON THE QUERY ----------------------------------------------------------------

# Query
# query = "What is the budget?"

# similar = vectorstore.similarity_search_with_score(query, k=2) # k=2 to return 2 similar documents

# for doc in similar:
#   print(doc)
#   print(f"Page: {doc[0].metadata['page']}, Similarity: {doc[1]}")



# PROMPT TEMPLATE ----------------------------------------------------------------------------------------

template = """Answer the question based on the follwing contex: {context}
If you are unable to find the answer within the context, please respond with 'I don't know'.

Question: {question}
"""

prompt = PromptTemplate(
  template = template,
  input_variables = ['context', 'question']
  )


# CHAIN -------------------------------------------------------------------------------------------------

# rag_chain = (
#   {"context": retriever, "question": RunnablePassthrough()}
#   | prompt
#   | model
#   | StrOutputParser()
# )
