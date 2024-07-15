import os
import pickle
import re
import time
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import FakeEmbeddings

# Function to clean text by removing all special characters and \n
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove non-alphanumeric characters
    # cleaned_text = re.sub(r'\n', '', cleaned_text)      # Remove \n characters
    # cleaned_text = re.sub(r'\n\n', '', cleaned_text)      # Remove \n characters
    # cleaned_text = re.sub(r'\xa0', '', cleaned_text)    # Remove non-breaking spaces
    return cleaned_text.strip()

# Load Google API Key
GOOGLE_API_KEY = st.secrets.secrets["GOOGLE_API_KEY"]

# Streamlit App
st.title("Retrieval-Based QA System")

# Sidebar for URL input and processing
st.sidebar.title("News Article URLs")
num_urls = st.sidebar.slider("Number of URLs", min_value=1, max_value=5, value=1)
urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i+1}")
    if url:
        urls.append(url)

# Initialize session state for vector index and docs
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "docs" not in st.session_state:
    st.session_state.docs = []

# Define file path for storing vector index
file_path = "vector_index.pkl"

if st.sidebar.button("Process URLs"):
    if urls:
        progress = st.sidebar.progress(0)
        step_count = 5

        # Step 1: Data Loading
        start_time = time.time()
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()
        elapsed_time = time.time() - start_time
        progress.progress(1 / step_count)
        st.sidebar.text(f"Data Loading... {elapsed_time:.2f} seconds ✅")

        # Step 2: Cleaning Data
        start_time = time.time()
        cleaned_data = []
        for doc in data:
            cleaned_content = clean_text(doc.page_content)
            cleaned_doc = Document(
                metadata=doc.metadata,
                page_content=cleaned_content
            )
            cleaned_data.append(cleaned_doc)
        elapsed_time = time.time() - start_time
        progress.progress(2 / step_count)
        st.sidebar.text(f"Cleaning Data... {elapsed_time:.2f} seconds ✅")

        # Step 3: Text Splitting
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ' '],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(cleaned_data)
        st.session_state.docs = docs  # Store docs in session state
        elapsed_time = time.time() - start_time
        progress.progress(3 / step_count)
        st.sidebar.text(f"Text Splitting... {elapsed_time:.2f} seconds ✅")

        # Step 4: Embedding
        start_time = time.time()
        embeddings = FakeEmbeddings(size=200)
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        st.session_state.vector_index = vectorindex_openai  # Store vector index in session state
        elapsed_time = time.time() - start_time
        progress.progress(4 / step_count)
        st.sidebar.text(f"Building Embedding Vector... {elapsed_time:.2f} seconds ✅")

        # Step 5: Saving Vector Index
        start_time = time.time()
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_openai, f)
        elapsed_time = time.time() - start_time
        progress.progress(5 / step_count)
        st.sidebar.text(f"Saving Vector Index... {elapsed_time:.2f} seconds ✅")

        st.success("URLs processed successfully.")
    else:
        st.error("Please enter at least one URL.")

# Main QA Interface
query = st.text_input("Enter your question:")
if query:
    if st.session_state.vector_index is not None:
        vectorstore = st.session_state.vector_index
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=GoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY, temperature=0.9), 
            retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)
        
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
    else:
        st.error("Please process the URLs first.")


# Display processed documents
if st.session_state.docs:
    st.header("Processed Documents")
    for i, doc in enumerate(st.session_state.docs):
        st.subheader(f"Document {i+1}")
        st.write(doc.page_content[:500] + "...")  # Display first 500 characters of each document