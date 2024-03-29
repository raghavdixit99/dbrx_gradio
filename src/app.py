def main():
    import streamlit as st
    from common import LangChainModel
    LM = LangChainModel()
    # from src.helper import DBRX
    # # This is the first API key input; no need to repeat it in the main function.
    # api_key = st.text_input(
    #     "Enter your Google API Key:", type="password", key="api_key_input"
    # )

    st.set_page_config(page_title="Document Genie", layout="wide")

    st.markdown("""
    ## Document Genie: Get instant insights from your Documents

    This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Databricks's DBRX-instruct and gpt4all embeddings. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

    ### How It Works

    Follow these simple steps to interact with the chatbot:

    1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

    2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
    """)


    st.header("AI clone chatbot💁")
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            key="pdf_uploader",
        )
        if st.button(
            "Submit & Process", key="process_button"
        ):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = LM.get_pdf_text(pdf_docs)
                text_chunks = LM.get_text_chunks(raw_text)

                st.session_state.vec_db = LM.get_vector_store(text_chunks)

                st.success("Done")

    user_question = st.text_input(
        "Ask a Question from the PDF Files", key="user_question"
    )
    
    if (
        user_question and st.session_state.vec_db
    ):  # Ensure vec_db is populated and user question are provided
        response = LM.user_input.remote(user_question, st.session_state.vec_db)
        st.write("Reply: ", response["output_text"])
    else:
        st.info("Please upload the documents and provide the question.")


if __name__ == "__main__":
    main()

