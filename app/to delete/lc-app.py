import os
import uuid
from dotenv import load_dotenv

from langfuse import Langfuse
# from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse
langfuse = Langfuse()

# Retrieve MODEL_NAME environment variable
model = os.getenv("MODEL_NAME")

trace_id = str(uuid.uuid4())
ollama_trace = langfuse.trace(
  id=trace_id,
  name='techspike_trace ollama llama3'
)

# Initialize Langfuse handler with corrected environment variable access
langfuse_handler = CallbackHandler(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# Initialize Ollama with the model name
llm = Ollama(model=model)

response = llm.invoke(
  [
    HumanMessage(content="Hi! I'm Barry"),
    AIMessage(content="Hello Barry! How can I assist you today?"),
    HumanMessage(content="What's my name?"),
  ],
  config={"callbacks": [langfuse_handler]})
  
response