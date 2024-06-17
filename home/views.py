from django.shortcuts import render
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from django.contrib import messages


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    analyze the provided context to answer the question. Include all relevant details and highlight their significance. If the answer can't be definitively found, offer a concise summary of the available information.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]
    return answer

def index(request):
    answer = ""
    file_uploaded = False
    if request.method == 'POST':
        pdf_docs = request.FILES.getlist('pdfInput')
        user_question = request.POST.get('questionInput')
        
        if pdf_docs:
            file_uploaded = True
            messages.success(request, 'File uploaded successfully.')
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            request.session['text_chunks'] = text_chunks
            get_vector_store(text_chunks)
        
        if 'text_chunks' in request.session and user_question:
            text_chunks = request.session['text_chunks']
            answer = user_input(user_question)
        
        return render(request, 'index.html', {'status': 'success', 'answer': answer, 'file_uploaded': file_uploaded})

    return render(request, 'index.html', {'status': 'fail', 'file_uploaded': file_uploaded}, status=400)