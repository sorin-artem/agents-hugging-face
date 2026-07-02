import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from openai import OpenAI

from tools import search_url, web_search


SEARCH_QUERY_PROMPT = """
You are helping answer a user question.
Create one concise web search query that finds the authoritative source page needed to answer it.
Do not copy the full question.
Remove unnecessary instruction wording.
Use keyword-style search, not a full sentence.
Keep the main entity name exactly.
Add the likely source page type when useful, such as discography, filmography, bibliography, official site, statistics, or archive.
If the question requests English Wikipedia, prefer a query that targets the relevant Wikipedia article title, for example:
Mercedes Sosa discography Wikipedia
Do not over-constrain the query with dates if that makes the source page harder to find.
Avoid queries that are likely to find copied Q&A pages, answer keys, or solution repositories.
Return only the search query, with no explanation.
"""

CHOOSE_URL_PROMPT = """
You are helping answer a user question.
Choose the single most useful URL from the web search results.
Prefer authoritative primary sources, official pages, and Wikipedia when requested.
Avoid copied Q&A pages, answer keys, solution repositories, and unrelated pages.
If no result is relevant to the main entity and question, return NO_RELEVANT_URL.
Return only the URL or NO_RELEVANT_URL, with no explanation.
"""

ANSWER_WITH_SEARCH_PROMPT = """
You are a general-purpose assistant.
Use the provided web search results and fetched page content to answer the user's question.
Return only the final answer.
Do not include explanations.
Do not write "Final Answer:".
If the evidence is insufficient, give your best concise answer.
"""


class GaiaState(TypedDict):
    question: str
    search_query: str
    search_results: str
    page_url: str
    page_content: str
    answer: str
    steps: list[str]


class GaiaAgent:
    def __init__(self) -> None:
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPEN_ROUTER_API_KEY.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2")
        self.search_max_tokens = int(
            os.getenv("OPENROUTER_SEARCH_MAX_TOKENS", "128"))
        self.answer_max_tokens = int(
            os.getenv("OPENROUTER_ANSWER_MAX_TOKENS", "1024"))
        self.graph = self._build_graph()

    def __call__(self, question: str) -> str:
        result = self.run(question)
        return result["answer"].strip()

    @traceable(name="gaia_agent_run")
    def run(self, question: str) -> GaiaState:
        initial_state: GaiaState = {
            "question": question,
            "search_query": "",
            "search_results": "",
            "page_url": "",
            "page_content": "",
            "answer": "",
            "steps": [],
        }
        return self.graph.invoke(initial_state)

    def _build_graph(self):
        workflow = StateGraph(GaiaState)

        workflow.add_node("create_search_query", self._create_search_query)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("choose_url", self._choose_url)
        workflow.add_node("fetch_url", self._fetch_url)
        workflow.add_node("answer", self._answer)

        workflow.add_edge(START, "create_search_query")
        workflow.add_edge("create_search_query", "web_search")
        workflow.add_edge("web_search", "choose_url")
        workflow.add_edge("choose_url", "fetch_url")
        workflow.add_edge("fetch_url", "answer")
        workflow.add_edge("answer", END)

        return workflow.compile()

    @traceable(name="create_search_query")
    def _create_search_query(self, state: GaiaState) -> dict[str, object]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SEARCH_QUERY_PROMPT},
                {"role": "user", "content": state["question"]},
            ],
            temperature=0,
            max_tokens=self.search_max_tokens,
        )

        query = (response.choices[0].message.content or "").strip()
        return {
            "search_query": query or state["question"],
            "steps": [*state["steps"], "create_search_query"],
        }

    @traceable(name="web_search")
    def _web_search(self, state: GaiaState) -> dict[str, object]:
        query = state["search_query"] or state["question"]
        return {
            "search_results": web_search(query),
            "steps": [*state["steps"], "web_search"],
        }

    @traceable(name="choose_url")
    def _choose_url(self, state: GaiaState) -> dict[str, object]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CHOOSE_URL_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{state['question']}\n\n"
                        f"Search query:\n{state['search_query']}\n\n"
                        f"Web search results:\n{state['search_results']}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=256,
        )

        url = (response.choices[0].message.content or "").strip()
        if url == "NO_RELEVANT_URL":
            url = ""

        return {
            "page_url": url,
            "steps": [*state["steps"], "choose_url"],
        }

    @traceable(name="fetch_url")
    def _fetch_url(self, state: GaiaState) -> dict[str, object]:
        return {
            "page_content": search_url(state["page_url"]),
            "steps": [*state["steps"], "fetch_url"],
        }

    @traceable(name="answer")
    def _answer(self, state: GaiaState) -> dict[str, object]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANSWER_WITH_SEARCH_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{state['question']}\n\n"
                        f"Web search query:\n{state['search_query']}\n\n"
                        f"Web search results:\n{state['search_results']}\n\n"
                        f"Fetched URL:\n{state['page_url']}\n\n"
                        f"Fetched page content:\n{state['page_content']}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=self.answer_max_tokens,
        )

        answer = response.choices[0].message.content
        return {
            "answer": answer.strip() if answer else "",
            "steps": [*state["steps"], "answer"],
        }
