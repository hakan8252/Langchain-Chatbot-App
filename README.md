# LangChain Chatbot - Retrieval-Based QA System
This Streamlit web application allows users to ask questions about news articles using a retrieval-based question-answering system. The application processes URLs of news articles, splits the text into manageable chunks, creates embeddings, and retrieves relevant information to answer user queries.

# Features
* URL Processing: Enter and process URLs of news articles to create a vector index for efficient retrieval.
* Question Answering: Ask questions about the processed news articles and get relevant answers along with sources.
* Default URL and Query: Provides a default URL and query for ease of use, which can be modified by the user.
* Delay Between Queries: Ensures a 5-second delay between successive queries to manage API usage efficiently.


## Getting Started
### Prerequisites
Before running the application, make sure you have the following installed:

* Python 3.x
* Streamlit
* langchain
* langchain-google-genai
* FAISS
* UnstructuredURLLoader

Installation
Clone the repository:

```bash
git clone https://github.com/hakan8252/Langchain-Chatbot-App.git
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Access the app in your web browser at http://localhost:8501.

Project Link: https://langchain-chatbot-llm.streamlit.app/

# Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.

# Acknowledgements
This project uses LangChain for the retrieval-based QA system.
Data processing is handled by UnstructuredURLLoader.
Vector embeddings are managed using FAISS.
Google Generative AI is used for the language model.
Feel free to adjust any part of this README to better fit your project specifics.
