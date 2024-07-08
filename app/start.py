import ollama
import uuid
import os
from langfuse import Langfuse
from dotenv import load_dotenv

langfuse = Langfuse()
load_dotenv()

# create a new trace
trace_id = str(uuid.uuid4())
ollama_trace = langfuse.trace(
  id=trace_id,
  name='techspike_trace'
)

# generation
user_prompt = 'What is pi?'
ollama_generation = ollama_trace.generation(
  name='ollma_generation',
  input={
    "prompt": user_prompt
    }
)


model_name = os.getenv('MODEL_NAME')

output = ollama.generate(model_name, prompt=user_prompt)
ollama_generation.end(
  output=output.get('response'),
  usage={
    'input': output.get('prompt_eval_count'),
    'output': output.get('eval_count')
    }
)

ollama_trace.update(
  input=user_prompt, 
  output=output
)