import modal

from modal import Image, Secret, Stub, method, enter

# vol = Volume.from_name("dbrx-huggingface-volume", create_if_missing=True)
# LANCEDB_URI = "/lancedb"

image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:23.10-py3", add_python="3.10")
    .apt_install("git", gpu="H100")
    .pip_install(
        "streamlit",
        "PyPDF2",
        "lancedb",
        "gpt4all",
        "langchain",
        "langchain-community",
        "pyarrow",
        "transformers>=4.39.2",
        "tiktoken>=0.6.0",
        "torch",
        "hf_transfer",
        "accelerate",
        gpu="H100",
    )
    .run_commands("echo $CUDA_HOME", "nvcc --version",force_build=True)
    # .run_commands(
    #     "python -m pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    # )
    # .run_commands(
    #     "git clone https://github.com/vllm-project/vllm.git",
    #     "python -m pip install -e vllm/.",
    #     gpu="H100",
    # )
    # .run_commands("python -m pip install --upgrade pip")
    # .run_commands("python -m pip install -U wheel setuptools")
    # .run_commands("python -m pip install -U flash-attn --no-build-isolation")
    .env({"CGO_ENABLED0": 0})
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

stub = Stub("dbrx_hf", image=image, secrets=[Secret.from_name("HF_token_raghav")])

GPU_CONFIG = modal.gpu.H100(count=6)

GPU_CONFIG_STD = modal.gpu.A100(count=1)

with image.imports():
    from typing import Any

    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import GPT4AllEmbeddings
    from langchain_community.vectorstores import LanceDB


@stub.cls(image=image)
class LangChainModel:

    vector_store = None
    embedder = GPT4AllEmbeddings()


    @method()
    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000
        )
        chunks = text_splitter.split_text(text)
        return chunks

    @method()
    def get_vector_store(self, text_chunks):

        if self.vector_store is not None:
            self.vector_store.add_texts(text_chunks)

        self.vector_store = LanceDB.from_texts(
            text_chunks, embedding=self.embedder
        )

    @method()
    def user_input(self, user_question):
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized")
        docs = self.vector_store.similarity_search(user_question)
        chain = self.get_conversational_chain.remote()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        return response

    @modal.build(gpu=GPU_CONFIG)
    @enter(gpu=GPU_CONFIG)
    def load(self):
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        import os

        hf_token = os.environ["HF_TOKEN"]

        tokenizer = AutoTokenizer.from_pretrained(
            "databricks/dbrx-instruct", trust_remote_code=True, token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            "databricks/dbrx-instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=hf_token,
        )
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        print("self.vec", self.vector_store, "embedder", self.embedder) 

        # gpu_llm = HuggingFacePipeline.from_model_id(
        # model_id="gpt2",
        # task="text-generation",
        # device=0,  # replace with device_map="auto" to use the accelerate library.
        # pipeline_kwargs={"max_new_tokens": 10},
        # )

    @method(gpu=GPU_CONFIG_STD)
    def get_conversational_chain(self, user_question):
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

        if self.vector_store is None:
            raise ValueError("Vector store is not initialized")
        docs = self.vector_store.similarity_search(user_question)
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        return response

