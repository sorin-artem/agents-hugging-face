import os
import asyncio
from pathlib import Path
import sys

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
from llama_index.core.tools import FunctionTool
from llama_index.tools.google import GmailToolSpec
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

load_dotenv()

INPUT_FILE = os.getenv("LLAMAINDEX_INPUT_FILE", "./2-razgovornik-dom.pdf")
CHROMA_DB_PATH = "./alfred_chroma_db"
CHROMA_COLLECTION = "alfred"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_MCP_SERVER = Path(__file__).with_name("demo_mcp_server.py")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
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


async def load_local_mcp_tools():
    """Expose MCP tools from a URL or from the bundled local server."""
    if MCP_SERVER_URL:
        mcp_client = BasicMCPClient(MCP_SERVER_URL)
    else:
        mcp_client = BasicMCPClient(sys.executable, args=[str(LOCAL_MCP_SERVER)])

    mcp_tool = McpToolSpec(client=mcp_client)
    return await mcp_tool.to_tool_list_async()


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

    def search_document(query: str) -> str:
        """Search the PDF and return the answer with source page metadata."""
        response = query_engine.query(query)
        lines = [f"Answer: {response.response}", "", "Sources:"]

        for source in response.source_nodes:
            metadata = source.node.metadata
            page = metadata.get("page_label", "unknown")
            file_name = metadata.get("file_name", "unknown")
            text = source.node.get_content().strip().replace("\n", " ")

            lines.append(f"- file: {file_name}, page_label: {page}, score: {source.score}")
            lines.append(f"  text: {text[:500]}")

        return "\n".join(lines)

    tool = FunctionTool.from_defaults(
        fn=search_document,
        name="search_document",
        description=(
            "Search the PDF document. Returns the answer, source text, "
            "file name, page_label metadata, and retrieval scores."
        ),
    )
    tool_spec = GmailToolSpec()
    gmail_tools = tool_spec.to_tool_list()
    mcp_tools = await load_local_mcp_tools()
    print(f"Loaded MCP tools: {[mcp_tool.metadata.name for mcp_tool in mcp_tools]}")

    tools = [tool, *mcp_tools]
    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=(
            "You are a helpful assistant. Use tools when needed. "
            "Write the final answer text in English."
        ),
    )
    agent_context = Context(agent)
    response = await agent.run(
        user_msg=(
            "Use the local MCP add tool to calculate 17 + 25. Use the local "
            "MCP current_time tool to get the current time. Use the local MCP "
            "echo tool with the message 'MCP tools are working'. Then use the "
            "search_document tool to find 'холодильник' in the document. "
            "Answer with the sum, the current time, the echoed message, "
            "the Hebrew translation, and the page number. "
            "For the page number, you must use page_label from the tool output. "
            "Write the final response in English."
        ),
        ctx=agent_context,
    )
    print(response)
    print("\nTOOL CALLS:")
    print(getattr(response, "tool_calls", None))

if __name__ == "__main__":
    asyncio.run(main())
