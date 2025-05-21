from django.shortcuts import render
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError # Import PdfReadError
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


def get_pdf_text(pdf_docs): # Expects a list, even if single
    text = ""
    # Assuming we're still focusing on the first PDF in the list for extraction,
    # as per current design where get_vector_store takes one filename.
    # This function can be refactored if multiple PDFs are to be processed into one text blob.
    if not pdf_docs:
        return None # Or raise error, or return ""
        
    pdf_file_obj = pdf_docs[0] # Process the first PDF object in the list
    try:
        pdf_reader = PdfReader(pdf_file_obj)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except PdfReadError:
        # Specific error for PyPDF2 related to reading issues (corruption, encryption)
        # Return None to signal an error to the caller for this specific PDF
        return None 
    except Exception:
        # Catch any other unexpected errors during PDF processing
        # Return None to signal an error
        return None

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, pdf_filename):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Ensure the faiss_index directory exists
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
        
    # Sanitize pdf_filename to create a valid directory name
    base_name = os.path.splitext(pdf_filename)[0]
    index_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_name) + "_index"
    
    vector_store.save_local(os.path.join("faiss_index", index_name))

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
    
    index_dir = "faiss_index"
    all_docs = []
    
    if os.path.exists(index_dir) and os.listdir(index_dir):
        faiss_indexes = [d for d in os.listdir(index_dir) if os.path.isdir(os.path.join(index_dir, d))]
        
        if not faiss_indexes:
            return "No PDF has been indexed yet. Please upload a PDF."

        # Load the first index
        first_index_name = faiss_indexes[0]
        # FAISS saves two files: index.faiss and index.pkl. We need to load using the directory name.
        current_db = FAISS.load_local(os.path.join(index_dir, first_index_name), embeddings, allow_dangerous_deserialization=True)

        # Merge other indexes if they exist
        if len(faiss_indexes) > 1:
            for index_name in faiss_indexes[1:]:
                try:
                    # FAISS.load_local expects the directory containing index.faiss and index.pkl
                    index_to_merge = FAISS.load_local(os.path.join(index_dir, index_name), embeddings, allow_dangerous_deserialization=True)
                    # The FAISS object itself is the index. Langchain's FAISS object has a `merge_from` method.
                    current_db.merge_from(index_to_merge)
                except Exception as e:
                    # Handle cases where a directory might not be a valid FAISS index
                    # Or if merge_from fails for some reason.
                    print(f"Could not merge index {index_name}: {e}")
                    # Optionally, continue without merging this specific index or collect docs separately
                    # For now, we'll just print the error and continue with the current_db
                    pass
        
        docs = current_db.similarity_search(user_question)
        if not docs:
             return "No relevant documents found for your question in the indexed PDFs."
    else:
        return "No PDF has been indexed yet. Please upload a PDF."

    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]
    return answer

def index(request):
    answer = ""
    # file_uploaded = False # No longer needed as Django messages are used
    if request.method == 'POST':
        pdf_docs = request.FILES.getlist('pdfInput')
        user_question = request.POST.get('questionInput')
        
        valid_pdf_docs = []
        if pdf_docs:
            for doc in pdf_docs:
                is_pdf_content_type = doc.content_type == 'application/pdf'
                is_pdf_extension = doc.name.lower().endswith('.pdf')
                
                if is_pdf_content_type and is_pdf_extension:
                    valid_pdf_docs.append(doc)
                else:
                    messages.error(request, f"Invalid file type: '{doc.name}'. Only PDF files are accepted.")
            
            if not valid_pdf_docs:
                # No valid PDFs found, render the page with error messages
                # Need to fetch indexed_files for the template even when returning early
                indexed_files = []
                if os.path.exists("faiss_index"):
                    for item_name in os.listdir("faiss_index"):
                        item_path = os.path.join("faiss_index", item_name)
                        if os.path.isdir(item_path):
                            original_filename = item_name.replace("_index", "") + ".pdf"
                            indexed_files.append(original_filename)
                return render(request, 'index.html', {'status': 'fail', 'answer': answer, 'indexed_files': indexed_files})

            # Process the first valid PDF
            first_valid_pdf = valid_pdf_docs[0]
            pdf_filename = first_valid_pdf.name # Get name before passing to get_pdf_text

            raw_text = get_pdf_text([first_valid_pdf]) # Pass as a list

            if raw_text is None:
                messages.error(request, f"Error processing PDF '{pdf_filename}': The file may be corrupted, password-protected, or unreadable.")
            else:
                # Successfully got text, now check if it's empty
                if not raw_text.strip():
                    messages.warning(request, f"PDF '{pdf_filename}' was processed, but no text could be extracted. The document might be image-based or empty.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, pdf_filename)
                    messages.success(request, f"PDF '{pdf_filename}' processed and indexed successfully.")

        if user_question:
            messages.info(request, f"Searching for answers to '{user_question}'...")
            answer = user_input(user_question)
            # Check content of answer to decide on message type
            if "No PDF has been indexed yet" in answer or "No relevant documents found" in answer:
                messages.warning(request, answer) # Use the answer itself as the warning
            elif answer:
                messages.success(request, "Answer found.")
            else: # Should not happen if user_input always returns a string
                messages.warning(request, "Could not find a definitive answer based on the provided documents.")

        indexed_files = []
        if os.path.exists("faiss_index"):
            for item_name in os.listdir("faiss_index"):
                item_path = os.path.join("faiss_index", item_name)
                if os.path.isdir(item_path):
                    original_filename = item_name.replace("_index", "") + ".pdf" # Reconstruct original filename
                    indexed_files.append(original_filename)
        
        return render(request, 'index.html', {'status': 'success', 'answer': answer, 'indexed_files': indexed_files}) # Removed file_uploaded

    # For GET request
    indexed_files = []
    if os.path.exists("faiss_index"):
        for item_name in os.listdir("faiss_index"):
            item_path = os.path.join("faiss_index", item_name)
            if os.path.isdir(item_path):
                original_filename = item_name.replace("_index", "") + ".pdf" # Reconstruct original filename
                indexed_files.append(original_filename)
    
    # The 'status' variable might be less relevant now.
    # Removed file_uploaded from context
    return render(request, 'index.html', {'status': 'fail', 'answer': answer, 'indexed_files': indexed_files}, status=400) # Removed file_uploaded