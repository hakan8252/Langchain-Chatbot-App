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
import langchain
from langchain_community.embeddings import FakeEmbeddings

# Load Google API Key
GOOGLE_API_KEY = st.secrets.secrets["GOOGLE_API_KEY"]

# Streamlit App
st.title("Retrieval-Based QA System")

# Sidebar for URL input and processing
st.sidebar.title("News Article URLs")
# Set the default URL as a placeholder
default_url = "https://www.euronews.com/my-europe/2024/07/16/two-far-right-groups-cordoned-off-from-power-roles-in-the-european-parliament"
num_urls = st.sidebar.slider("Number of URLs", min_value=1, max_value=5, value=1)
urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}", value=default_url if i == 0 else "", key=f"url_{i+1}")
    if url:
        urls.append(url)

# Define file path for storing vector index
file_path = "vector_index.pkl"

if "docs" not in st.session_state:
    st.session_state.docs = None

if st.sidebar.button("Process URLs"):
    if urls:
        progress = st.sidebar.progress(0)
        step_count = 4

        # Step 1: Data Loading
        start_time = time.time()
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()
        elapsed_time = time.time() - start_time
        progress.progress(1 / step_count)
        st.sidebar.text(f"Data Loading... {elapsed_time:.2f} seconds ✅")
        time.sleep(4)

        # Step 3: Text Splitting
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ' '],
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(data)
        st.session_state.docs = docs  # Store docs in session state
        elapsed_time = time.time() - start_time
        progress.progress(2 / step_count)
        st.sidebar.text(f"Text Splitting... {elapsed_time:.2f} seconds ✅")
        time.sleep(6)

        # Step 4: Embedding
        start_time = time.time()
        embeddings = FakeEmbeddings(size=200)
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        elapsed_time = time.time() - start_time
        progress.progress(3 / step_count)
        st.sidebar.text(f"Building Embedding Vector... {elapsed_time:.2f} seconds ✅")
        time.sleep(6)

        # Step 5: Saving Vector Index
        start_time = time.time()
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_openai, f)
        elapsed_time = time.time() - start_time
        progress.progress(4 / step_count)
        st.sidebar.text(f"Saving Vector Index... {elapsed_time:.2f} seconds ✅")

        st.success("URLs processed successfully.")
    else:
        st.error("Please enter at least one URL.")


default_query = "What is the main context of the text?"

# Main QA Interface
query = st.text_input("Enter your question:", value=default_query)
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=GoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY, temperature=0.5), 
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
            
            # Countdown timer
            remaining_time = 5
            with st.empty():
                while remaining_time > 0:
                    st.write(f"Next query available in {remaining_time} seconds.")
                    time.sleep(1)
                    remaining_time -= 1
    else:
        st.error("Vector index file does not exist. Please process the URLs first.")

# Display processed documents
# st.header("Processed Documents")
# Access docs from session state
# docs = st.session_state.get("docs", None)
# if docs:
#     for i, doc in enumerate(docs):
#         st.subheader(f"Document {i+1}")
#         st.write(doc.page_content[:500] + "...")  # Display first 500 characters of each document
# else:
#     st.text("There are no documents")