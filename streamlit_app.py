import os
import pickle
import re
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
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)     # Remove \n characters
    cleaned_text = re.sub(r'\xa0', ' ', cleaned_text)   # Remove non-breaking spaces
    return cleaned_text.strip()

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

if st.sidebar.button("Process URLs"):
    if urls:
        # Define URL Loader
        loaders = UnstructuredURLLoader(urls=urls)
        
        # Load Data
        data = loaders.load()
        
        # Clean the content of each document
        cleaned_data = []
        for doc in data:
            cleaned_content = clean_text(doc.page_content)
            cleaned_doc = Document(
                metadata=doc.metadata,
                page_content=cleaned_content
            )
            cleaned_data.append(cleaned_doc)
        
        # Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ' '],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(cleaned_data)
        
        # Define Embeddings
        embeddings = FakeEmbeddings(size=200)
        
        # Create FAISS Vector Index
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        
        # Store vector index locally
        file_path = "vector_index.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_openai, f)
        
        # Load vector index from file
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorIndex = pickle.load(f)
        
        # Create QA Chain
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        llm = GoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY, temperature=0.9)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
        
        st.session_state["chain"] = chain
        st.success("URLs processed successfully.")
    else:
        st.error("Please enter at least one URL.")

# Main QA Interface
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if query:
        if "chain" in st.session_state:
            chain = st.session_state["chain"]
            result = chain({"question": query}, return_only_outputs=False)
            st.write("**Answer:**")
            st.write(result['answer'])
            st.write("**Sources:**")
            for source in result['sources']:
                st.write(source)
        else:
            st.error("Please process the URLs first.")
    else:
        st.write("Please enter a question.")
