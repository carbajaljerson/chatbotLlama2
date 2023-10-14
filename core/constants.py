import os

MODEL_PATH = "./llm_model/llama-2-7b-chat.ggmlv3.q4_0.bin"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '161b38d5-c986-4da9-ac94-d980a3fa0de7')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')
INDEX_NAME = "langchain-pinecone-llama2"