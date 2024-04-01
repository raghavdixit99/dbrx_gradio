import modal

from modal import Image, Secret, Stub, Volume, method, enter
from pathlib import Path

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
        "tiktoken",
        "cupy-cuda11x",
        gpu="H100",
    )
    .run_commands("echo $CUDA_HOME", "nvcc --version")
    # .run_commands(
    #     "python -m pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    # )
    .run_commands(
        "git clone https://github.com/vllm-project/vllm.git",
        "python -m pip install -e vllm/.",
        gpu="H100",
    )
    .run_commands("python -m pip install --upgrade pip")
    .run_commands("python -m pip install -U wheel setuptools")
    .run_commands("python -m pip install -U flash-attn --no-build-isolation")
    .env({"CGO_ENABLED0": 0})
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .env({"CUDA_LAUNCH_BLOCKING": "1"})
    .run_commands("pip freeze | grep vllm")
)

stub = Stub("dbrx_demo", image=image, secrets=[Secret.from_name("HF_token_raghav")])

GPU_CONFIG = modal.gpu.H100(count=8)

with image.imports():
    from typing import Any

    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import GPT4AllEmbeddings
    from langchain_community.vectorstores import LanceDB


@stub.cls(image=image, gpu=GPU_CONFIG)
class LangChainModel:
    # @method()
    # def get_pdf_text(self, pdf_docs):
    #     text = ""
    #     for pdf in pdf_docs:
    #         pdf_reader = PdfReader(pdf)
    #         for page in pdf_reader.pages:
    #             text += page.extract_text()
    #     return text

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

        embedding = GPT4AllEmbeddings()
        tbl = self._init_table(embedding)
        self.vector_store = LanceDB.from_texts(
            text_chunks, embedding=embedding, connection=tbl
        )

    # @method()
    # def user_input(self, user_question):
    #     docs = self.vector_store.similarity_search(user_question)
    #     chain = self.get_conversational_chain.remote()
    #     response = chain(
    #         {"input_documents": docs, "question": user_question},
    #         return_only_outputs=True,
    #     )
    #     return response

    @modal.build()
    @enter()
    def load(self):
        from langchain_community.llms import VLLM
        import torch

        torch.cuda.empty_cache()

        # Patch issue from https://github.com/vllm-project/vllm/issues/1116
        import ray
        ray.shutdown()
        ray.init(num_gpus=GPU_CONFIG.count)

        self.vector_store = None
        self.llm = VLLM(
            model="databricks/dbrx-instruct",
            tensor_parallel_size=8,
            # vllm_kwargs={"quantization": "awq"},  # no support for quantization dbrx
            max_new_tokens=200,
            trust_remote_code=True,
        )

        # gpu_llm = HuggingFacePipeline.from_model_id(
        # model_id="gpt2",
        # task="text-generation",
        # device=0,  # replace with device_map="auto" to use the accelerate library.
        # pipeline_kwargs={"max_new_tokens": 10},
        # )

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

    def _init_table(self, embeddings) -> Any:
        import lancedb
        import pyarrow as pa

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
        db = lancedb.connect("/lancedb")
        tbl = db.create_table("vectorstore", schema=schema, mode="overwrite")
        return tbl


# @stub.local_entrypoint()
# def main():
#     # download_models.remote()

#     import subprocess

#     cmd = "python3 -m pip freeze | grep transformers"
#     subprocess.run(cmd, shell=True)
#     # pkgs_str = ",".join(pkgs)

#     # pk
#     # print("model weights downloaded..")

#     # LM = Cls.lookup("dbrx_llm", "LangChainModel")
#     # langchain_class = LM()
#     # print("langchain class object :", langchain_class)
# @modal.build()
# def download_model(self):
#     import os
#     import torch
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     from transformers.utils import move_cache

#     # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#     hf_token = os.environ["HF_TOKEN"]
#     torch.cuda.empty_cache()
#     # device = torch.device("cuda")

#     if hf_token is None:
#         raise ValueError("HF_HUB_TOKEN environment variable is not set.")

#     model_id = "databricks/dbrx-instruct"
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_id,
#         trust_remote_code=True,
#         token=hf_token,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map='auto',
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         token=hf_token,
#     )
#     tokenizer.save_pretrained(f"{MODEL_DIR}/tokenizer")
#     model.to('cpu').save_pretrained(f"{MODEL_DIR}/llm")
#     move_cache()
# mandatory for hf models


# if os.environ.get("CUDA_VISIBLE_DEVICES") != '5':
#     os.environ["CUDA_VISIBLE_DEVICES"] = '5'

# # quantization_config = QuantoConfig(weights="int8")
# device = "auto"

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_ID, use_fast=True, use_cache=True, cache_dir=CACHE_PATH, device_map=device, trust_remote_code=True , local_files_only = True
# )
# model = AutoModelForCausalLM.from_pretrained(
#     CACHE_PATH, use_cache=True, trust_remote_code=True, local_files_only = True , low_cpu_mem_usage = True
# )

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device_map=device,
#     pipeline_kwargs={"max_new_tokens": 200},
# )

# self.llm = HuggingFacePipeline(pipeline=pipe)
# .run_commands(" export CUDA_HOME=/usr/local/cuda ")
# .run_commands(
#     "/usr/local/cuda/bin/nvcc --version","export FORCE_CUDA='1'"
# )
# .run_commands("export PATH=/usr/local/cuda/bin:$PATH ",
#     "export CUDACXX=/usr/local/cuda/bin/nvcc",
#               """export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" """,
#               "export FORCE_CMAKE=1"
# )

# .run_commands(
#     "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
#     "dpkg -i cuda-keyring_1.1-1_all.deb",
#     "apt-get update",
#     "apt-get -y install cuda-toolkit-12-4",
#     gpu="H100",
# )


# volume = Volume.from_name("dbrx-huggingface-volume", create_if_missing=True)

# REMOTE_PATH = "/root"


# CACHE_PATH = "/weights"

# def download_model(self):
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     import torch
#     import os

#     # os.environ["CUDA_VISIBLE_DEVICES"] = '5'

#     print("downloading model..")

#     hf_token = os.environ["HF_TOKEN"]
#     # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     device_map = "auto"
#     torch.cuda.empty_cache()

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         use_cache=True,
#         device_map=device_map,
#         trust_remote_code=True,
#         token=hf_token,
#         torch_dtype=torch.bfloat16,
#     )

#     model.save_pretrained(CACHE_PATH, safe_serialization=True)

#     print('model weights downloaded..', os.listdir(CACHE_PATH) )

#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_ID, use_fast=True, use_cache=True, token=hf_token, trust_remote_code=True
#     )

#     tokenizer.save_pretrained(CACHE_PATH, safe_serialization=True)

#     print('tokenizer weights downloaded..', os.listdir(CACHE_PATH) )
# .run_commands("nvcc --version")
# .run_commands('${CUDA_HOME}/bin/nvcc --version')
# .run_commands(
#     "python -m cupyx.tools.install_library --cuda 12.x --library cutensor"
# )
# .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
# "hf_transfer",
# "git+https://github.com/huggingface/accelerate",
# "git+https://github.com/huggingface/transformers"
