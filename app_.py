import os
from PyPDF2 import PdfReader
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from typing import Any, List

from modal import (
    Image,
    Stub,
    asgi_app,
    enter,
    method,
)

image = (
    Image.from_registry("langchain/langchain")
    .pip_install(
        "hf_transfer",
        "PyPDF2",
        "lancedb",
        "transformers",
        "tiktoken",
        "gpt4all",
        "langchain-community",
        "pyarrow",
        "accelerate",
        "fastapi",
        "gradio",
    )
    .run_commands("apt-get update && apt-get install ffmpeg libsm6 libxext6  -y")
    .env({"CGO_ENABLED0": 0})
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

web_app = FastAPI()
stub = Stub(name="dbrx-demo-2", image=image)
FASTAI_HOME = "/fastai_home"
os.environ["FASTAI_HOME"] = FASTAI_HOME

# assets_path = pathlib.Path("/tmp/gradio/").resolve()
# streamlit_script_mount = Mount.from_local_dir(assets_path, remote_path="/data")


@stub.cls(image=image, gpu="a100")
class LangChainModel:
    def __init__(self):
        self.pdf_docs = None
        self.vec_db = None
        self.user_question = None
        self.response = None
        self.llm = None

    def get_pdf_text(self, pdf) -> str:
        text = ""
        pdf_reader = PdfReader(pdf.name)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def get_text_chunks(self, text) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks) -> Any:
        embedding = GPT4AllEmbeddings()
        tbl = self._init_table(embedding)
        vector_store = LanceDB.from_texts(
            text_chunks, embedding=embedding, connection=tbl
        )
        self.vec_db = vector_store

    def user_input(self, user_question):
        if not self.vec_db:
            raise ValueError("Please provide a PDF file first")

        docs = self.vec_db.similarity_search(user_question)
        chain = self.get_conversational_chain.remote()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        return response

    @enter()
    def load(self):
        # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from langchain_community.llms import HuggingFaceEndpoint

        model_id = "databricks/dbrx-instruct"

        gpu_llm = HuggingFaceEndpoint(
            repo_id=model_id,
            task="question-answering",
            huggingfacehub_api_token = ""  # replace with device_map="auto" to use the accelerate library.
        )

        self.llm = gpu_llm

        # I also tried the below code but it didn't work:
        # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        # import torch

        # model_id = "databricks/dbrx-instruct"
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_id,
        #     trust_remote_code=True,
        #     token="hf_token",
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True,
        #     token="hf_token",
        # )

        # pipe = pipeline(
        #     "question-answering", model=model, tokenizer=tokenizer, max_new_tokens=200
        # )
        # self.llm = HuggingFacePipeline(pipeline=pipe)


    @method()
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

    # def run(self):
    #     if self.pdf_docs:
    #         raw_text = self.get_pdf_text(self.pdf_docs)
    #         text_chunks = self.get_text_chunks(raw_text)
    #         self.vec_db = self.get_vector_store(text_chunks)

    #     if self.user_question and self.vec_db:
    #         self.response = self.user_input(self.user_question, self.vec_db)
    #     return self.response

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


@stub.function(
    image=image,
    # mounts=[Mount.from_local_dir(assets_path, remote_path="/")],
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    LM = LangChainModel()

    def chatbot_app(pdf_file, question):
        # if not pdf_file:
        #     raise ValueError("Please upload a PDF file")
        # if not question:
        #     raise ValueError("Please provide a question before hitting Submit")

        text = ""
        print("Reading PDF file...")
        print("PDF file path: ", pdf_file)

        text = LM.get_pdf_text(pdf_file)

        print("Done reading PDF file, got text - ", text[:5])

        chunks = LM.get_text_chunks(text)

        LM.get_vector_store(chunks)

        print("Done with vector store creation: ", LM.vec_db)

        return None

    def get_response(query):
        response = LM.user_input(query)
        return response

    if_1 = gr.Interface(
        fn=chatbot_app,
        inputs=[gr.File(label="Upload PDF")],
        outputs=None,
        title="PDF Contextual RAG Chatbot",
        description="Upload a PDF to provide context for the chatbot. Then ask your question and click on Submit!",
    )

    if_2 = gr.Interface(
        fn=get_response,
        inputs=[gr.Textbox(label="Your Question")],
        outputs="text",
        title="Question Answering Chatbot",
        description="RAG chat bot",
    )

    final_interface = gr.TabbedInterface([if_1, if_2], ["upload", "Chat"])

    return mount_gradio_app(
        app=web_app,
        blocks=final_interface,
        path="/",
    )
