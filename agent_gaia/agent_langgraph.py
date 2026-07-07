import os
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from openai import OpenAI

from tools import AGENT_TOOLS


ACTION_PROMPT = """
You are a general-purpose assistant that can use tools.
Choose the next action needed to answer the user's question.

Rules:
- If the question can be answered directly from the text, use final_answer.
- Use web_search when external information is needed.
- Use search_url only for authoritative, relevant URLs from prior observations.
- Avoid copied Q&A pages, answer keys, solution repositories, and unrelated pages.
- If search results are poor, use web_search again with a better keyword query.
- Use exactly one tool call.
- final_answer.answer must contain only the answer, not a sentence and not JSON.
"""

def _openrouter_tool_schema(tool_name: str) -> dict[str, object]:
    langchain_tool = AGENT_TOOLS[tool_name]
    return {
        "type": "function",
        "function": {
            "name": langchain_tool.name,
            "description": langchain_tool.description,
            "parameters": langchain_tool.args_schema.model_json_schema(),
        },
    }


ACTION_TOOLS = [
    _openrouter_tool_schema("web_search"),
    _openrouter_tool_schema("search_url"),
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Finish with the concise final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Only the final answer, with no explanation.",
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
]


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
            tools=ACTION_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        action, action_input = self._parse_model_message(message)
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

    def _parse_model_message(self, message: object) -> tuple[str, dict[str, str]]:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            tool_call = tool_calls[0]
            function = getattr(tool_call, "function", None)
            name = str(getattr(function, "name", "")).strip()
            raw_arguments = str(getattr(function, "arguments", "{}"))
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            return self._normalize_action(name, arguments)

        raw_action = getattr(message, "content", None) or ""
        return self._parse_action(str(raw_action))

    def _parse_action(self, raw_action: str) -> tuple[str, dict[str, str]]:
        try:
            parsed = json.loads(raw_action)
        except json.JSONDecodeError:
            return "final_answer", {"answer": raw_action.strip()}

        action = str(parsed.get("action", "")).strip()
        action_input = parsed.get("action_input", {})
        if not isinstance(action_input, dict):
            action_input = {}

        return self._normalize_action(action, action_input)

    def _normalize_action(self, action: str, action_input: dict) -> tuple[str, dict[str, str]]:
        if action not in {*AGENT_TOOLS.keys(), "final_answer"}:
            return "final_answer", {"answer": ""}

        normalized_input = {str(key): str(value) for key, value in action_input.items()}
        if action == "final_answer":
            answer = normalized_input.get("answer", "").strip()
            if answer.startswith("{") and answer.endswith("}"):
                nested_action, nested_input = self._parse_action(answer)
                if nested_action != "final_answer" or nested_input.get("answer") != answer:
                    return nested_action, nested_input
            normalized_input["answer"] = answer

        return action, normalized_input

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
            observation = AGENT_TOOLS[action].invoke({"query": query})
            update["search_query"] = query
            update["search_results"] = observation
        elif action == "search_url":
            url = action_input.get("url", "").strip()
            observation = AGENT_TOOLS[action].invoke({"url": url})
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

