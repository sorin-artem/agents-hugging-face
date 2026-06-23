import random
from datetime import date

from huggingface_hub import list_models
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import StructuredTool, Tool

from rag_agent.retriever import extract_text


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


def get_random_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


def get_current_date() -> str:
    """Returns today's date."""
    return date.today().isoformat()


search_tool = DuckDuckGoSearchRun()


def get_latest_news(topic: str) -> str:
    """Searches the web for the latest news about a topic."""
    today = date.today().isoformat()
    query = f"{topic} latest news {today}"
    return search_tool.run(query)


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


guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation.",
)

guest_conversation_starter_tool = Tool(
    name="guest_conversation_starter",
    func=start_conversation,
    description=(
        "Generates a polite conversation starter for a gala guest based on their "
        "name, relation, description, or other guest information."
    ),
)

weather_tool = Tool(
    name="weather_info_retriever",
    func=get_random_weather_info,
    description="Fetches dummy weather information for a given location.",
)

current_date_tool = StructuredTool.from_function(
    func=get_current_date,
    name="current_date",
    description="Returns today's current date in ISO format. Use this for questions about 'today', 'now', 'right now', or current events.",
)

latest_news_tool = Tool(
    name="latest_news_retriever",
    func=get_latest_news,
    description=(
        "Searches the web for the latest news about a topic. Use this for current "
        "events, recent announcements, breaking news, or questions asking what is "
        "happening right now."
    ),
)

hub_tool = Tool(
    name="hub_stats_retriever",
    func=get_hub_stats,
    description=(
        "Searches Hugging Face Hub models by keyword and returns the top results "
        "sorted by downloads. Use this only when the user explicitly asks about "
        "popular Hugging Face models, coding models, or model download stats."
    ),
)

tools = [
    guest_info_tool,
    guest_conversation_starter_tool,
    search_tool,
    weather_tool,
    current_date_tool,
    latest_news_tool,
    hub_tool,
]
