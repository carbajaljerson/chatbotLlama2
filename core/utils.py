# Importing langchai and llama2 related functions
from langchain.llms import CTransformers
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.vectorstores import Pinecone

# Importing general libraries
import os
import pinecone


# Constants
MODEL_PATH = "./llm_model/llama-2-7b-chat.ggmlv3.q4_0.bin"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'f875819a-13c8-493c-a063-8a8dcce9a986')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')
INDEX_NAME = "langchain-pinecone-llama2"


def pineconeLogin():

    """
    Connect to the Pinecone service using the provided API key and environment.

    This function initializes a connection to the Pinecone service with the specified
    API key and environment settings.

    Args:
        None

    Returns:
        None

    Note:
        Make sure to set the 'PINECONE_API_KEY' and 'PINECONE_API_ENV' environment variables
        with your Pinecone API key and environment settings before calling this function.
    """

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
        )

# Functions without a page
def createLlm(MODEL_PATH):

    """
    Creates and initializes a language model using the CTransformers library.

    This function loads a pre-trained language model from the specified MODEL_PATH
    and configures it for generating new text based on the 'llama' model type.

    Args:
    - MODEL_PATH (str): The path to the pre-trained language model checkpoint that
      you want to use. This checkpoint should be compatible with the CTransformers
      library.

    Returns:
    - llm: An instance of the CTransformers language model initialized with the
      specified model checkpoint and settings. You can use this model to generate
      text by calling its methods.

    Note:
    - Ensure that the MODEL_PATH points to a valid pre-trained model checkpoint.
    """

    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=4096, 
        temperature=0.01,
    )

    print('Model loaded')

    return llm

def createEmbeddings(EMBEDDING_MODEL):

    """
    Create embeddings using the Sentence Transformers model.

    This function initializes a Sentence Transformers model with the specified `EMBEDDING_MODEL` and returns it.

    Args:
        EMBEDDING_MODEL (str): The name or path of the Sentence Transformers model to be used for embedding generation.
            You can specify a pre-trained model's name (e.g., 'bert-base-nli-mean-tokens') or a path to a
            custom model if available.

    Returns:
        embeddings (SentenceTransformerEmbeddings): An instance of the SentenceTransformerEmbeddings class
            initialized with the specified model. This object can be used to generate embeddings for input text.
    """

    #Create embeddings here
    print("Loading Sentence Transformers Model")
    embeddings = SentenceTransformerEmbeddings(
                                model_name=EMBEDDING_MODEL,
                                model_kwargs={"device": "cpu"})
    
    print("Finished loading sentence transformers model")

    return embeddings

def readVectorStore(INDEX_NAME, embeddings):

    """
    Creates a Pinecone index from a pre-trained embedding model.

    This function takes an embedding model and converts it into a Pinecone index
    that can be used for similarity search.

    Args:
        EMBEDDING_MODEL (object): An instance of a pre-trained embedding model.
            This model should provide a method to read vector embeddings from
            the data.

    Returns:
        PineconeIndex: A Pinecone index object that is created from the provided
        EMBEDDING_MODEL. This index can be used to perform similarity searches on
        the embeddings.
    """

    return Pinecone.from_existing_index(INDEX_NAME, embeddings)

def answer(query, llm, vectorStore):

    """
    Generates an answer to a given question using a question-answering chain.

    This function takes a user query, a question-answering language model (llm), 
    and a vectorStore for document retrieval. It first loads a question-answering
    chain of the specified type (e.g., "stuff"). Then, it performs document similarity
    search to find relevant documents for the query. Finally, it runs the loaded 
    question-answering chain to generate an answer to the input question.

    Args:
        query (str): The user's question for which an answer is to be generated.
        llm: The question-answering language model used for generating the answer.
        vectorStore: The vector store for document retrieval and similarity search.

    Returns:
        str: The generated answer to the user's query.

    Note:
        This function relies on pre-configured question-answering chains and
        a vectorStore for document retrieval.
    """

    print('Generating Answer')
    chain = load_qa_chain(llm, chain_type="stuff")

    print('Generating Chain')
    docs= vectorStore.similarity_search(query, k=1)

    output = chain.run(input_documents=docs,
              question=query)
    
    return output