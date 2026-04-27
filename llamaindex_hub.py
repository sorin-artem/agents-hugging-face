import os
import asyncio
from pathlib import Path

import chromadb
import llama_index.core
from dotenv import load_dotenv
from phoenix.otel import register
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.evaluation import FaithfulnessEvaluator

load_dotenv()

INPUT_FILE = os.getenv("LLAMAINDEX_INPUT_FILE", "./2-razgovornik-dom.pdf")
CHROMA_DB_PATH = "./alfred_chroma_db"
CHROMA_COLLECTION = "alfred"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT",
    os.getenv("PHOENIX_ENDPOINT", "https://app.phoenix.arize.com"),
).removesuffix("/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "default")


def configure_phoenix_tracing() -> None:
    if not PHOENIX_API_KEY:
        print("PHOENIX_API_KEY is not set. Skipping Phoenix tracing.")
        return

    os.environ["PHOENIX_API_KEY"] = PHOENIX_API_KEY
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = PHOENIX_COLLECTOR_ENDPOINT

    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME,
    )
    llama_index.core.set_global_handler(
        "arize_phoenix",
        tracer_provider=tracer_provider,
    )


def load_documents(input_file: str) -> list[Document]:
    if os.path.isfile(input_file):
        reader = PDFReader()
        return reader.load_data(file=Path(input_file))

    print(f"Input file not found: {input_file!r}. Using Document.example().")
    return [Document.example()]


async def main() -> None:
    configure_phoenix_tracing()

    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() > 0:
        print(
            f"Collection {CHROMA_COLLECTION!r} already has "
            f"{chroma_collection.count()} items. Skipping ingestion."
        )
    else:
        documents = load_documents(INPUT_FILE)

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                HuggingFaceEmbedding(model_name=EMBED_MODEL),
            ],
            vector_store=vector_store,
        )

        stored_nodes = await pipeline.arun(documents=documents)
        print(
            f"Stored {len(stored_nodes)} nodes in Chroma collection "
            f"{CHROMA_COLLECTION!r} at {CHROMA_DB_PATH!r}."
        )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model)

    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
        similarity_top_k=10

    )
    response = query_engine.query(
        "on which page i can find translation of the word 'key'?",)
    print(response)
    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_result = evaluator.evaluate_response(response=response)
    print(eval_result.passing)

if __name__ == "__main__":
    asyncio.run(main())
