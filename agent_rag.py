from huggingface_hub import list_models
import random
from datetime import date
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain_core.tools import StructuredTool, Tool
from langchain_community.retrievers import BM25Retriever
import datasets
from langchain_core.documents import Document
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# Load the dataset
guest_dataset = datasets.load_dataset(
    "agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]


bm25_retriever = BM25Retriever.from_documents(docs)


def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."


guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)


def start_conversation(guest_name: str) -> str:
    """Generates a conversation starter for a given guest based on their information."""
    guest_info = extract_text(guest_name)
    if guest_info == "No matching guest information found.":
        return guest_info

    return (
        f"Guest information:\n{guest_info}\n\n"
        f"Conversation starter:\n"
        f"Start by greeting {guest_name}, then mention one specific detail from their "
        "background and ask a polite follow-up question about it."
    )


guest_conversation_starter_tool = Tool(
    name="guest_conversation_starter",
    func=start_conversation,
    description=(
        "Generates a polite conversation starter for a gala guest based on their "
        "name, relation, description, or other guest information."
    ),
)

chat = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

search_tool = DuckDuckGoSearchRun()


def get_random_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


weather_tool = Tool(
    name="weather_info_retriever",
    func=get_random_weather_info,
    description="Fetches dummy weather information for a given location."
)


def get_current_date() -> str:
    """Returns today's date."""
    return date.today().isoformat()


current_date_tool = StructuredTool.from_function(
    func=get_current_date,
    name="current_date",
    description="Returns today's current date in ISO format. Use this for questions about 'today', 'now', 'right now', or current events.",
)


def get_latest_news(topic: str) -> str:
    """Searches the web for the latest news about a topic."""
    today = date.today().isoformat()
    query = f"{topic} latest news {today}"
    return search_tool.run(query)


latest_news_tool = Tool(
    name="latest_news_retriever",
    func=get_latest_news,
    description=(
        "Searches the web for the latest news about a topic. Use this for current "
        "events, recent announcements, breaking news, or questions asking what is "
        "happening right now."
    ),
)


def get_hub_stats(query: str) -> str:
    """Searches popular Hugging Face Hub models by keyword."""
    try:
        models = list(list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=5,
        ))

        if not models:
            return f"No models found on Hugging Face Hub for query: {query}."

        lines = [
            f"Top Hugging Face Hub models for query '{query}', sorted by downloads:"
        ]
        for index, model in enumerate(models, start=1):
            downloads = model.downloads or 0
            lines.append(f"{index}. {model.id} with {downloads:,} downloads")

        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching models for query '{query}': {str(e)}"


hub_tool = Tool(
    name="hub_stats_retriever",
    func=get_hub_stats,
    description=(
        "Searches Hugging Face Hub models by keyword and returns the top results "
        "sorted by downloads. Use this for questions about popular Hugging Face "
        "models, coding models, or model download stats."
    )
)

tools = [guest_info_tool, guest_conversation_starter_tool,
         search_tool, weather_tool, current_date_tool, latest_news_tool, hub_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


builder = StateGraph(AgentState)


# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(
    content="Great Tesla with latest news about Elon Musk")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print("\nExecution trace:")
for message in response["messages"]:
    message.pretty_print()
