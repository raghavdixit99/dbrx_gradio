from modal import Stub, Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from typing import Any
from serve_streamlit import streamlit_script_mount
from modal import (
    Image,
    Mount,
    Secret,
    Stub,
    Volume,
    asgi_app,
    enter,
    method,
)

image = (
    Image.from_registry("langchain/langchain")
    .pip_install(
        "streamlit",
        "hf_transfer",
        "PyPDF2",
        "lancedb",
        "transformers",
        "tiktoken",
        "gpt4all",
        "langchain-community",
        "pyarrow",
        "accelerate",
    )
    .run_commands("apt-get update && apt-get install ffmpeg libsm6 libxext6  -y")
    .env({"CGO_ENABLED0": 0})
)

stub = Stub(name="dbrx-demo-2", image=image)

@stub.cls(image=image, gpu = 'any', mounts=[streamlit_script_mount])
class LangChainModel:

    def __init__(self):
        self.pdf_docs = None
        self.vec_db = None
        self.user_question = None
        self.response = None
        self.llm=None

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        embedding = GPT4AllEmbeddings()
        tbl = self._init_table(embedding)
        vector_store = LanceDB.from_texts(
            text_chunks, embedding=embedding, connection=tbl
        )
        return vector_store


    def user_input(self, user_question, vec_db):
        docs = vec_db.similarity_search(user_question)
        chain = self.get_conversational_chain.remote()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        return response
    
    @enter
    def load(self):
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_id = "databricks/dbrx-instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token="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",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token="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",
        )

        pipe = pipeline(
            "question-answering", model=model, tokenizer=tokenizer, max_new_tokens=200
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    @method
    def get_conversational_chain(self):
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context but I can provide you with..", and then search your knowledge base to give RELEVANT answers ONLY, don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain


    def run(self):
        if self.pdf_docs:
            raw_text = self.get_pdf_text(self.pdf_docs)
            text_chunks = self.get_text_chunks(raw_text)
            self.vec_db = self.get_vector_store(text_chunks)

        if self.user_question and self.vec_db:
            self.response = self.user_input(self.user_question, self.vec_db)
        return self.response
    

    def _init_table(self, embeddings) -> Any:
        import pyarrow as pa
        import lancedb

        schema = pa.schema(
            [
                pa.field(
                    "vector",
                    pa.list_(
                        pa.float32(),
                        len(embeddings.embed_query("test")),  # type: ignore
                    ),
                ),
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
            ]
        )
        db = lancedb.connect("/tmp/lancedb")
        tbl = db.create_table("vectorstore", schema=schema, mode="overwrite")
        return tbl
    


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=10000, chunk_overlap=1000
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embedding = GPT4AllEmbeddings()
#     tbl = _init_table(embedding)
#     vector_store = LanceDB.from_texts(
#         text_chunks, embedding=embedding, connection=tbl
#     )
#     return vector_store

# def user_input(user_question, vec_db):
#     docs = vec_db.similarity_search(user_question)
#     chain = get_conversational_chain.remote()
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True,
#     )
#     return response

# @stub.function(gpu="any",mounts=[streamlit_script_mount])
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context but I can provide you with..", and then search your knowledge base to give RELEVANT answers ONLY, don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
#     from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#     from langchain.chains.question_answering import load_qa_chain
#     from langchain.prompts import PromptTemplate
#     import torch

#     model_id = "databricks/dbrx-instruct"
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_id,
#         trust_remote_code=True,
#         token="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",
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         token="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",
#     )

#     pipe = pipeline(
#         "question-answering", model=model, tokenizer=tokenizer, max_new_tokens=200
#     )
#     llm = HuggingFacePipeline(pipeline=pipe)

#     prompt = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
#     chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
#     return chain
