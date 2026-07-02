import os
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from openai import OpenAI

from tools import search_url, web_search


ACTION_PROMPT = """
You are a general-purpose assistant that can use tools.
Choose the next action needed to answer the user's question.

Available actions:
- web_search: search the web for relevant sources. Input: {"query": "..."}
- search_url: read a specific URL. Input: {"url": "..."}
- final_answer: finish with a concise answer. Input: {"answer": "..."}

Rules:
- If the question can be answered directly from the text, use final_answer.
- Use web_search when external information is needed.
- Use search_url only for authoritative, relevant URLs from prior observations.
- Avoid copied Q&A pages, answer keys, solution repositories, and unrelated pages.
- If search results are poor, use web_search again with a better keyword query.
- Return only valid JSON with keys "action" and "action_input".
- Do not include markdown or explanations.

Example:
{"action": "web_search", "action_input": {"query": "Mercedes Sosa discography Wikipedia"}}
"""


class GaiaState(TypedDict):
    question: str
    action: str
    action_input: dict[str, str]
    observations: list[dict[str, str]]
    search_query: str
    search_results: str
    page_url: str
    page_content: str
    answer: str
    steps: list[str]
    step_count: int


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
        self.action_max_tokens = int(
            os.getenv("OPENROUTER_ACTION_MAX_TOKENS", "512"))
        self.answer_max_tokens = int(
            os.getenv("OPENROUTER_ANSWER_MAX_TOKENS", "1024"))
        self.max_steps = int(os.getenv("AGENT_MAX_STEPS", "8"))
        self.graph = self._build_graph()

    def __call__(self, question: str) -> str:
        result = self.run(question)
        return result["answer"].strip()

    @traceable(name="gaia_agent_run")
    def run(self, question: str) -> GaiaState:
        initial_state: GaiaState = {
            "question": question,
            "action": "",
            "action_input": {},
            "observations": [],
            "search_query": "",
            "search_results": "",
            "page_url": "",
            "page_content": "",
            "answer": "",
            "steps": [],
            "step_count": 0,
        }
        return self.graph.invoke(initial_state)

    def _build_graph(self):
        workflow = StateGraph(GaiaState)

        workflow.add_node("decide_action", self._decide_action)
        workflow.add_node("execute_tool", self._execute_tool)

        workflow.add_edge(START, "decide_action")
        workflow.add_conditional_edges(
            "decide_action",
            self._route_after_decision,
            {
                "execute_tool": "execute_tool",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "execute_tool",
            self._route_after_tool,
            {
                "decide_action": "decide_action",
                "end": END,
            },
        )

        return workflow.compile()

    @traceable(name="decide_action")
    def _decide_action(self, state: GaiaState) -> dict[str, object]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ACTION_PROMPT},
                {
                    "role": "user",
                    "content": self._format_action_context(state),
                },
            ],
            temperature=0,
            max_tokens=self.action_max_tokens,
        )

        raw_action = response.choices[0].message.content or ""
        action, action_input = self._parse_action(raw_action)
        return {
            "action": action,
            "action_input": action_input,
            "answer": action_input.get("answer", "") if action == "final_answer" else state["answer"],
            "steps": [*state["steps"], f"decide_action:{action}"],
        }

    def _format_action_context(self, state: GaiaState) -> str:
        observations = json.dumps(
            state["observations"],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"Question:\n{state['question']}\n\n"
            f"Previous observations:\n{observations}\n\n"
            f"Steps used: {state['step_count']}/{self.max_steps}"
        )

    def _parse_action(self, raw_action: str) -> tuple[str, dict[str, str]]:
        try:
            parsed = json.loads(raw_action)
        except json.JSONDecodeError:
            return "final_answer", {"answer": raw_action.strip()}

        action = str(parsed.get("action", "")).strip()
        action_input = parsed.get("action_input", {})
        if not isinstance(action_input, dict):
            action_input = {}

        if action not in {"web_search", "search_url", "final_answer"}:
            return "final_answer", {"answer": ""}

        return action, {str(key): str(value) for key, value in action_input.items()}

    def _route_after_decision(self, state: GaiaState) -> str:
        if state["action"] == "final_answer":
            return "end"
        if state["step_count"] >= self.max_steps:
            return "end"
        return "execute_tool"

    def _route_after_tool(self, state: GaiaState) -> str:
        if state["answer"]:
            return "end"
        if state["step_count"] >= self.max_steps:
            return "end"
        return "decide_action"

    @traceable(name="execute_tool")
    def _execute_tool(self, state: GaiaState) -> dict[str, object]:
        action = state["action"]
        action_input = state["action_input"]
        observation = ""
        update: dict[str, object] = {
            "step_count": state["step_count"] + 1,
            "steps": [*state["steps"], f"execute_tool:{action}"],
        }

        if action == "web_search":
            query = action_input.get("query", "").strip() or state["question"]
            observation = web_search(query, max_results=8)
            update["search_query"] = query
            update["search_results"] = observation
        elif action == "search_url":
            url = action_input.get("url", "").strip()
            observation = search_url(url)
            update["page_url"] = url
            update["page_content"] = observation
        else:
            observation = f"Unknown action: {action}"

        update["observations"] = [
            *state["observations"],
            {
                "action": action,
                "action_input": json.dumps(action_input, ensure_ascii=False),
                "observation": observation,
            },
        ]
        return update

