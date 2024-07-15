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
    cleaned_text = re.sub(r'\n', '', cleaned_text)      # Remove \n characters
    cleaned_text = re.sub(r'\xa0', '', cleaned_text)    # Remove non-breaking spaces
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

# Placeholder for main content
main_placeholder = st.empty()

# Define file path for storing vector index
file_path = "vector_index.pkl"

if st.sidebar.button("Process URLs"):
    if urls:
        main_placeholder.text("Data Loading...Started...✅✅✅")
        
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
        main_placeholder.text("Text Splitting...Started...✅✅✅")
        docs = text_splitter.split_documents(cleaned_data)
        
        # Define Embeddings
        embeddings = FakeEmbeddings(size=200)
        
        # Create FAISS Vector Index
        main_placeholder.text("Building Embedding Vector...✅✅✅")
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_openai, f)

        st.success("URLs processed successfully.")
    else:
        st.error("Please enter at least one URL.")

# Main QA Interface
query = main_placeholder.text_input("Enter your question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
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
