import os
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.messages import get_messages
from unittest.mock import patch, MagicMock, mock_open

from PyPDF2.errors import PdfReadError
# Ensure your view functions are correctly imported.
# If views.py is in the same directory as tests.py (inside 'home' app):
from .views import get_pdf_text, get_vector_store, user_input 

# If your project structure is different, adjust the import path.
# e.g., from home.views import get_pdf_text ...


class GetPdfTextTests(TestCase):
    @patch('home.views.PdfReader')
    def test_get_pdf_text_success(self, mock_pdf_reader_class):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is page text. "
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader_instance.pages = [mock_page, mock_page]
        mock_pdf_reader_class.return_value = mock_pdf_reader_instance

        # Create a mock file object. It needs a 'name' attribute for some internal Django checks if passed as UploadedFile.
        # For get_pdf_text, it's passed directly, so a simple MagicMock is fine if it's only read.
        mock_pdf_file = MagicMock(spec=SimpleUploadedFile) # Using spec to be more file-like
        mock_pdf_file.name = "test.pdf" 
        
        text = get_pdf_text([mock_pdf_file]) # Pass as a list
        self.assertEqual(text, "This is page text. This is page text. ")
        mock_pdf_reader_class.assert_called_once_with(mock_pdf_file)

    @patch('home.views.PdfReader')
    def test_get_pdf_text_read_error(self, mock_pdf_reader_class):
        mock_pdf_reader_class.side_effect = PdfReadError("Mocked PDF Read Error")
        mock_pdf_file = MagicMock(spec=SimpleUploadedFile)
        mock_pdf_file.name = "error.pdf"
        
        text = get_pdf_text([mock_pdf_file])
        self.assertIsNone(text)

    @patch('home.views.PdfReader')
    def test_get_pdf_text_empty_extraction(self, mock_pdf_reader_class):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "" # Empty text
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader_class.return_value = mock_pdf_reader_instance
        
        mock_pdf_file = MagicMock(spec=SimpleUploadedFile)
        mock_pdf_file.name = "empty.pdf"
        
        text = get_pdf_text([mock_pdf_file])
        self.assertEqual(text, "")

    def test_get_pdf_text_no_docs(self):
        text = get_pdf_text([]) # Pass an empty list
        self.assertIsNone(text) # Or "" depending on implementation, current views.py returns None


class IndexViewFileHandlingTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Attempt to reverse the URL. This requires URL configuration to be set up.
        # If 'home' is an app and urls.py has `app_name = 'home'`, use 'home:index'.
        # If 'home.urls' is included in project urls.py without namespace, 'index' is fine.
        try:
            self.index_url = reverse('index') 
        except Exception as e:
            # Fallback or skip tests if URL reversing fails.
            # This often happens if urls.py is not correctly set up or not found by the test runner.
            # For now, let's assume a simple path if reverse fails, common in minimal setups.
            self.index_url = '/' # Adjust if your app is not at root.
            print(f"Warning: URL reversing failed with {e}. Using fallback URL '{self.index_url}'. Ensure your URLs are configured.")


    @patch('home.views.get_vector_store')
    @patch('home.views.get_text_chunks')
    @patch('home.views.get_pdf_text')
    def test_index_view_valid_pdf_upload(self, mock_get_pdf_text, mock_get_text_chunks, mock_get_vector_store):
        mock_get_pdf_text.return_value = "This is pdf text."
        mock_get_text_chunks.return_value = ["chunk1", "chunk2"]
        
        pdf_content = b"%PDF-1.4\n%%EOF" # Minimal valid PDF structure for some checks
        pdf_file = SimpleUploadedFile("test.pdf", pdf_content, content_type="application/pdf")
        
        with patch('os.path.exists') as mock_os_exists: # Mock os.path.exists for listing indexed_files
            mock_os_exists.return_value = False # No existing faiss_index
            response = self.client.post(self.index_url, {'pdfInput': [pdf_file], 'questionInput': ''})
        
        self.assertEqual(response.status_code, 200)
        messages = list(get_messages(response.wsgi_request)) # Use get_messages
        self.assertTrue(any(f"PDF 'test.pdf' processed and indexed successfully." in str(msg) for msg in messages))
        mock_get_pdf_text.assert_called_once()
        # Check the actual file object passed to get_pdf_text
        self.assertEqual(mock_get_pdf_text.call_args[0][0][0].name, "test.pdf")
        mock_get_text_chunks.assert_called_once_with("This is pdf text.")
        mock_get_vector_store.assert_called_once_with(["chunk1", "chunk2"], "test.pdf")

    def test_index_view_invalid_file_type(self):
        txt_file = SimpleUploadedFile("test.txt", b"this is not a pdf", content_type="text/plain")
        with patch('os.path.exists') as mock_os_exists:
            mock_os_exists.return_value = False
            response = self.client.post(self.index_url, {'pdfInput': [txt_file], 'questionInput': ''})
        
        self.assertEqual(response.status_code, 200) 
        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any("Invalid file type: 'test.txt'. Only PDF files are accepted." in str(msg) for msg in messages))

    @patch('home.views.get_pdf_text')
    def test_index_view_pdf_read_error(self, mock_get_pdf_text):
        mock_get_pdf_text.return_value = None 
        
        pdf_file = SimpleUploadedFile("corrupted.pdf", b"%PDF-...", content_type="application/pdf")
        with patch('os.path.exists') as mock_os_exists:
            mock_os_exists.return_value = False
            response = self.client.post(self.index_url, {'pdfInput': [pdf_file], 'questionInput': ''})
        
        self.assertEqual(response.status_code, 200)
        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any("Error processing PDF 'corrupted.pdf': The file may be corrupted, password-protected, or unreadable." in str(msg) for msg in messages))

    @patch('home.views.get_text_chunks') 
    @patch('home.views.get_pdf_text')
    def test_index_view_pdf_empty_text(self, mock_get_pdf_text, mock_get_text_chunks):
        mock_get_pdf_text.return_value = "   " 
        
        pdf_file = SimpleUploadedFile("empty.pdf", b"%PDF-...", content_type="application/pdf")
        with patch('os.path.exists') as mock_os_exists:
            mock_os_exists.return_value = False
            response = self.client.post(self.index_url, {'pdfInput': [pdf_file], 'questionInput': ''})
        
        self.assertEqual(response.status_code, 200)
        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any("PDF 'empty.pdf' was processed, but no text could be extracted." in str(msg) for msg in messages))
        mock_get_text_chunks.assert_not_called()


class GetVectorStoreTests(TestCase):
    @patch('home.views.FAISS') 
    @patch('home.views.GoogleGenerativeAIEmbeddings') 
    @patch('os.makedirs') 
    @patch('os.path.exists') 
    def test_get_vector_store_saves_correctly(self, mock_path_exists, mock_makedirs, mock_embeddings_class, mock_faiss_class):
        mock_path_exists.return_value = False 
        mock_embeddings_instance = mock_embeddings_class.return_value
        mock_vector_store_instance = mock_faiss_class.from_texts.return_value
        
        text_chunks = ["chunk1", "chunk2"]
        pdf_filename = "my test document.pdf"
        
        get_vector_store(text_chunks, pdf_filename)
        
        mock_path_exists.assert_called_once_with("faiss_index")
        mock_makedirs.assert_called_once_with("faiss_index")
        mock_embeddings_class.assert_called_once_with(model="models/embedding-001")
        mock_faiss_class.from_texts.assert_called_once_with(text_chunks, mock_embeddings_instance)
        
        expected_sanitized_dirname = "my_test_document_index"
        expected_save_path = os.path.join("faiss_index", expected_sanitized_dirname)
        mock_vector_store_instance.save_local.assert_called_once_with(expected_save_path)

class UserInputTests(TestCase):
    @patch('home.views.FAISS')
    @patch('home.views.GoogleGenerativeAIEmbeddings')
    @patch('home.views.load_qa_chain')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    def test_user_input_no_indexes_found(self, mock_isdir, mock_listdir, mock_path_exists, mock_load_qa_chain, mock_embeddings_class, mock_faiss_class):
        mock_path_exists.return_value = True 
        mock_listdir.return_value = [] 
        mock_embeddings_instance = mock_embeddings_class.return_value
        
        result = user_input("What is AI?")
        self.assertEqual(result, "No PDF has been indexed yet. Please upload a PDF.")
        mock_embeddings_class.assert_called_once_with(model="models/embedding-001")


    @patch('home.views.FAISS')
    @patch('home.views.GoogleGenerativeAIEmbeddings')
    @patch('home.views.load_qa_chain')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    def test_user_input_one_index(self, mock_isdir, mock_listdir, mock_path_exists, mock_load_qa_chain, mock_embeddings_class, mock_faiss_class):
        mock_path_exists.return_value = True
        mock_listdir.return_value = ["doc1_index"]
        mock_isdir.return_value = True

        mock_embeddings_instance = mock_embeddings_class.return_value
        mock_db_instance = mock_faiss_class.load_local.return_value
        mock_db_instance.similarity_search.return_value = ["doc_content"]
        
        mock_chain_instance = mock_load_qa_chain.return_value
        mock_chain_instance.invoke.return_value = {"output_text": "This is the answer."}

        result = user_input("What is AI?")
        
        self.assertEqual(result, "This is the answer.")
        mock_embeddings_class.assert_called_once_with(model="models/embedding-001")
        mock_faiss_class.load_local.assert_called_once_with(os.path.join("faiss_index", "doc1_index"), mock_embeddings_instance, allow_dangerous_deserialization=True)
        mock_db_instance.similarity_search.assert_called_once_with("What is AI?")
        mock_load_qa_chain.assert_called_once()

    @patch('home.views.FAISS')
    @patch('home.views.GoogleGenerativeAIEmbeddings')
    @patch('home.views.load_qa_chain')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    def test_user_input_multiple_indexes_merge(self, mock_isdir, mock_listdir, mock_path_exists, mock_load_qa_chain, mock_embeddings_class, mock_faiss_class):
        mock_path_exists.return_value = True
        mock_listdir.return_value = ["doc1_index", "doc2_index"]
        mock_isdir.return_value = True

        mock_embeddings_instance = mock_embeddings_class.return_value
        mock_db1_instance = MagicMock()
        mock_db2_instance = MagicMock()
        
        mock_faiss_class.load_local.side_effect = [mock_db1_instance, mock_db2_instance]
        mock_db1_instance.similarity_search.return_value = ["doc_content"] 

        mock_chain_instance = mock_load_qa_chain.return_value
        mock_chain_instance.invoke.return_value = {"output_text": "Merged answer."}

        result = user_input("What is AI from merged docs?")
        
        self.assertEqual(result, "Merged answer.")
        mock_embeddings_class.assert_called_once_with(model="models/embedding-001")
        self.assertEqual(mock_faiss_class.load_local.call_count, 2)
        mock_db1_instance.merge_from.assert_called_once_with(mock_db2_instance)
        mock_db1_instance.similarity_search.assert_called_once_with("What is AI from merged docs?")

    @patch('home.views.FAISS')
    @patch('home.views.GoogleGenerativeAIEmbeddings')
    @patch('home.views.load_qa_chain')
    @patch('os.path.exists')
    def test_user_input_faiss_index_dir_not_exists(self, mock_path_exists, mock_load_qa_chain, mock_embeddings_class, mock_faiss_class):
        mock_path_exists.return_value = False 
        mock_embeddings_instance = mock_embeddings_class.return_value
        
        result = user_input("What is AI?")
        self.assertEqual(result, "No PDF has been indexed yet. Please upload a PDF.")
        mock_embeddings_class.assert_called_once_with(model="models/embedding-001")
        mock_faiss_class.load_local.assert_not_called()
        mock_load_qa_chain.assert_not_called()

# Reminder for running tests:
# 1. Make sure 'home' app is in INSTALLED_APPS in your Django settings.
# 2. Make sure you have a urls.py in 'home' app and it's included in project's urls.py.
#    home/urls.py example:
#    from django.urls import path
#    from . import views
#    app_name = 'home' # Optional, but good practice for namespacing if you use it in reverse()
#    urlpatterns = [
#        path('', views.index, name='index'), # Ensure name='index'
#    ]
# 3. Run with: python manage.py test home
#
# The use of `get_messages(response.wsgi_request)` is a reliable way to access messages in tests.
# `SimpleUploadedFile` is suitable for mocking file uploads.
# Patching is targeted at `home.views.<object_name>` assuming that's how they are imported and used.
# For example, `from PyPDF2 import PdfReader` in views.py makes `home.views.PdfReader` the correct patch target.
# `os` functions are patched directly as they are globally available.
# Added `test_get_pdf_text_no_docs` for completeness.
# Corrected `mock_pdf_reader` to `mock_pdf_reader_class` for clarity as it's a class being mocked.
# Corrected `mock_embeddings` to `mock_embeddings_class` and `mock_embeddings_instance` for clarity.
# Similarly for `mock_faiss_class` and `mock_faiss_instance` (though not explicitly renamed, the usage is clearer).
# Used `get_messages` from `django.contrib.messages` for accessing messages, which is standard.
# Added `mock_os_exists.return_value = False` within `test_index_view_valid_pdf_upload` and similar tests to ensure `indexed_files` list is empty and doesn't interfere with message checking.
# Ensured `get_pdf_text` is called with a list: `get_pdf_text([mock_pdf_file])`.
# Ensured that the file object passed to `get_pdf_text` in `test_index_view_valid_pdf_upload` is checked.
# Added spec to mock_pdf_file for `GetPdfTextTests` for robustness.
# Added `app_name='home'` example in URL comments. If using namespace, `reverse('home:index')`. Current tests assume 'index' is globally unique or non-namespaced.
# For `IndexViewFileHandlingTests.setUp`, added a print warning if `reverse` fails and a fallback URL.
# This helps in environments where URL setup might be minimal or different.
# The `mock_os_exists` patch in `IndexViewFileHandlingTests` is to control the flow for `indexed_files` list population,
# ensuring that part of the view code doesn't cause issues or unexpected behavior during these specific unit tests.
# It's set to `False` to simulate no existing "faiss_index" directory when listing files for the template.
# If that part were crucial to these tests, `mock_listdir` and `mock_isdir` would also be needed for those specific tests.
# However, these tests focus on file processing messages, so neutralizing that part is fine.

# Final check on patch targets:
# - `home.views.PdfReader`: Correct, as `from PyPDF2 import PdfReader` is in `home/views.py`.
# - `home.views.get_vector_store`, `home.views.get_text_chunks`, `home.views.get_pdf_text`: Correct for testing `index` view by mocking other view functions.
# - `home.views.FAISS`, `home.views.GoogleGenerativeAIEmbeddings`, `home.views.load_qa_chain`: Correct as these are imported like `from ... import ...` in `home/views.py`.
# - `os.makedirs`, `os.path.exists`, `os.listdir`, `os.path.isdir`: Correct for mocking `os` module functions.
# All seems consistent.

```
