# Importing DJango libraries
from .models import *
from .forms import SignUpForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader, TextLoader , Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from django.views.decorators.http import require_http_methods
from langchain.vectorstores import Pinecone


# Importing constants
import core.constants as constant

# Importing utils
import core.utils as util
import importlib
importlib.reload(util)
       
   
# Create your views here.
def frontpage (request):

    return render(request, 'core/frontpage.html')

def signup(request):

    if request.method == 'POST':
        form = SignUpForm(request.POST)
        
        if form.is_valid():
            user = form.save()
            
            login(request, user)
            
            return redirect('frontpage')
    else:
        form = SignUpForm()
        
    return render(request, 'core/signup.html', {'form': form})

@login_required
def chat(request):
    queries = Userquery.objects.all().order_by('-id')[:5]
    context = {
        'queries': queries
    }
    return render(request, 'core/chat.html',context)       


@login_required
def CargaDocumental(request):
    context ={}
    if request.method == "POST":
        uploaded_file= request.FILES["document"]
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
    return render(request, 'core/CargaDocumental.html', context)       


def AI_GGML(request):

    """
    This view function handles user queries and leverages AI models to generate responses.

    Args:
        request (HttpRequest): The HTTP request object containing user input in the 'query' parameter.

    Returns:
        HttpResponse: A response object that renders the 'core/chat.html' template with context data, including:
            - 'queries': The latest 5 user queries from the database.
            - 'query': The user's input query.
            - 'output': The AI-generated response to the user's query.

    Note:
        This function expects the 'query' parameter to be present in the request's GET parameters. It uses the Userquery
        model to store the user's query and the corresponding AI-generated response in the database for future reference.
    """
    
    query = request.GET['query']
    
    
    util.pineconeLogin()
    #docs=util.ingest()
    
    llm = util.createLlm(constant.MODEL_PATH)
    embeddings = util.createEmbeddings(constant.EMBEDDING_MODEL)
    #vectorStore = util.readVectorStore(docs, constant.INDEX_NAME, embeddings)
    vectorStore = util.readVectorStore(constant.INDEX_NAME, embeddings)
    output = util.answer(query, llm, vectorStore)
    
    queries = Userquery.objects.all().order_by('id')[:5]
    
    #saving the query and output to database
    queryData = Userquery(
        query=query,
        reply=output
    )
    queryData.save() 
    context = {
        'queries':queries,
        'query':query,
        'output':output
    }
    
    return render(request, 'core/chat.html', context)


#python manage.py runserver