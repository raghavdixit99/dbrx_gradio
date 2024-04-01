import streamlit as st
from PyPDF2 import PdfReader
import modal


st.set_page_config(page_title="Document Genie", layout="wide")

st.header("AI clone chatbot💁")

st.markdown("""
This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Databricks's DBRX-instruct and gpt4all embeddings. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

@st.cache_resource
def fetch_modal_model():
    LM = modal.Cls.lookup("dbrx_hf", "LangChainModel")
    langchain_model = LM()
    return langchain_model


with st.sidebar:
    modal_model = fetch_modal_model()
    st.title("Menu:")
    pdf_docs = st.file_uploader(
        "Upload your PDF Files and Click on the Submit & Process Button",
        accept_multiple_files=True,
        key="pdf_uploader",
        type="pdf"
    )
    if st.button(
        "Submit & Process", key="process_button"
    ):  
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            st.write("Text extracted from the PDF files!")
            text_chunks = modal_model.get_text_chunks.remote(raw_text)
            modal_model.get_vector_store.remote(text_chunks)
            st.success("Done")

user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

if (
    user_question
):  # Ensure vec_db is populated and user question are provided
    response = modal_model.get_conversational_chain.remote(user_question)
    st.write("Reply: ", response["output_text"])
else:
    st.info("Please upload the documents and provide the question.")
