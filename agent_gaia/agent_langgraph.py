import asyncio
import os
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from openai import AsyncOpenAI

from tools import AGENT_TOOLS, ASYNC_AGENT_TOOLS


ACTION_PROMPT = """
You are the router for a general-purpose tool-using assistant.
Choose the next action needed to gather what is required to answer the question.

Rules:
- If the observations already contain enough information to answer, use final_answer.
- A separate reasoning step computes the final answer, so you do not need to solve the task yourself.
- Use web_search when external information is needed.
- Use search_url only for authoritative, relevant URLs from prior observations.
- If an attached file has already been successfully analyzed and the question does not explicitly require external/current/web information, use final_answer instead of web_search.
- Use run_python_attachment only when the attached file is a Python source file ending in .py and the task asks for its output or behavior.
- Use analyze_document_attachment when the task has an attached PDF, spreadsheet, CSV, JSON, or text document.
- Use analyze_image_attachment when the task has an attached image or asks about a picture, visual scene, or diagram.
- When using run_python_attachment, pass the task_id and file_name if present.
- When using analyze_document_attachment, pass the task_id, file_name if present, and the user's full document question.
- When using analyze_image_attachment, pass the task_id, file_name if present, and the user's full visual question.
- Never call run_python_attachment for .xlsx, .xlsm, .pdf, .csv, .json, image, or audio files.
- If image observations are already available and sufficient, use final_answer instead of calling analyze_image_attachment again.
- Avoid copied Q&A pages, answer keys, solution repositories, and unrelated pages.
- If search results are poor, use web_search again with a better keyword query.
- Use exactly one tool call.
- final_answer.answer is optional; if you fill it, include only a short draft answer, never a sentence or JSON.
"""

REASONING_PROMPT = """
You are a general-purpose reasoning agent.
Given a question and the observations gathered by tools, work out the answer yourself.

Rules:
- Think step by step and show your reasoning explicitly.
- Base every step only on the question and the observations; never invent facts.
- Trust successful observations over tool errors; a later tool error does not invalidate earlier useful data.
- If the observations are not enough to answer, state clearly what is missing instead of guessing.
"""

FINAL_ANSWER_PROMPT = """
You convert a reasoning trace into the final answer.

Rules:
- Output only the final answer to the original question, on a single line and nothing else. No explanation, citations, JSON, labels, or markdown.
- Follow exactly the format the question requests (for example a single number, a name, or algebraic notation).
- For numbers, do not use thousands separators: write 89706.00, not 89,706.00, unless the question explicitly asks for them.
- Base the answer strictly on the provided reasoning. Do not redo, extend, or second-guess the calculation, and never write filler like "Let me", "I need to", or "from where the reasoning left off".
- If the reasoning is cut off just before the last trivial step, finish that step silently and output only the resulting value.
- If the reasoning concludes the task cannot be answered from the available information, return exactly: Unable to determine
"""


class AgentError(RuntimeError):
    """Raised when a required model step fails or returns no usable output."""


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
    *[
        _openrouter_tool_schema(tool_name)
        for tool_name in AGENT_TOOLS
    ],
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Stop gathering information; a reasoning step will produce the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Optional short draft answer, with no explanation.",
                    },
                },
                "required": [],
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
    draft_answer: str
    reasoning: str
    search_query: str
    search_results: str
    page_url: str
    page_content: str
    answer: str
    steps: list[str]
    step_count: int


class GaiaAgent:
    def __init__(
        self,
        model: str | None = None,
        action_max_tokens: int | None = None,
        answer_max_tokens: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPEN_ROUTER_API_KEY.")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model or os.getenv(
            "OPENROUTER_MODEL", "minimax/minimax-m3")
        # Omitting max_tokens does NOT mean unlimited: the provider then applies its
        # own small default and truncates long reasoning mid-sentence (e.g. the
        # spreadsheet task got cut before finishing the last column). So we set a
        # generous explicit ceiling by default; pass "none"/"unlimited" in the env
        # to opt out and rely on the provider default instead.
        self.action_max_tokens = self._resolve_max_tokens(
            action_max_tokens, "OPENROUTER_ACTION_MAX_TOKENS", default=3072)
        self.answer_max_tokens = self._resolve_max_tokens(
            answer_max_tokens, "OPENROUTER_ANSWER_MAX_TOKENS", default=12000)
        self.max_steps = max_steps or int(os.getenv("AGENT_MAX_STEPS", "8"))
        # Some reasoning models can loop forever in their hidden reasoning
        # channel on hard tasks and never emit visible content, burning
        # the whole token budget for an empty answer. Keeping the hidden channel
        # off makes the model reason in plain text (which we already ask for) and
        # stop cleanly. Set OPENROUTER_REASONING=on to re-enable the hidden channel.
        self.reasoning_enabled = os.getenv(
            "OPENROUTER_REASONING", "off").strip().lower() in {
                "1", "true", "on", "yes", "enabled"}
        self.graph = self._build_graph()

    @staticmethod
    def _resolve_max_tokens(value: int | None, env_name: str, default: int | None) -> int | None:
        if value is not None:
            return value
        raw = os.getenv(env_name, "").strip().lower()
        if raw == "":
            return default
        if raw in {"0", "none", "unlimited", "-1"}:
            return None
        try:
            return int(raw)
        except ValueError:
            return default

    def __call__(self, question: str) -> str:
        result = self.run(question)
        return result["answer"].strip()

    @traceable(name="gaia_agent_run")
    def run(self, question: str) -> GaiaState:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(question))
        raise RuntimeError("GaiaAgent.run() cannot be called from an active event loop. Use arun().")

    @traceable(name="gaia_agent_arun")
    async def arun(self, question: str) -> GaiaState:
        return await self.graph.ainvoke(self._initial_state(question))

    def _initial_state(self, question: str) -> GaiaState:
        return {
            "question": question,
            "action": "",
            "action_input": {},
            "observations": [],
            "draft_answer": "",
            "reasoning": "",
            "search_query": "",
            "search_results": "",
            "page_url": "",
            "page_content": "",
            "answer": "",
            "steps": [],
            "step_count": 0,
        }

    def _build_graph(self):
        workflow = StateGraph(GaiaState)

        workflow.add_node("decide_action", self._decide_action)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("reason", self._reason)

        workflow.add_edge(START, "decide_action")
        workflow.add_conditional_edges(
            "decide_action",
            self._route_after_decision,
            {
                "execute_tool": "execute_tool",
                "reason": "reason",
            },
        )
        workflow.add_conditional_edges(
            "execute_tool",
            self._route_after_tool,
            {
                "decide_action": "decide_action",
                "reason": "reason",
            },
        )
        workflow.add_edge("reason", END)

        return workflow.compile()

    async def _chat(self, messages: list[dict[str, object]], *, max_tokens: int | None, step_name: str, **kwargs):
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if not self.reasoning_enabled:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("reasoning", {"enabled": False})
            kwargs["extra_body"] = extra_body
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                **kwargs,
            )
        except Exception as error:
            raise AgentError(
                f"{step_name} failed: model request error: {type(error).__name__}: {error}"
            ) from error

    async def _complete_text(self, system_prompt: str, user_prompt: str, *, max_tokens: int, step_name: str) -> str:
        response = await self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            step_name=step_name,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise AgentError(
                f"{step_name} failed: model {self.model!r} returned no text output."
            )
        return content

    @traceable(name="decide_action")
    async def _decide_action(self, state: GaiaState) -> dict[str, object]:
        response = await self._chat(
            [
                {"role": "system", "content": ACTION_PROMPT},
                {"role": "user",
                    "content": self._format_action_context(state)},
            ],
            max_tokens=self.action_max_tokens,
            step_name="Action decision step",
            tools=ACTION_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        action, action_input = self._parse_model_message(message)
        attachment_actions = {
            "run_python_attachment",
            "analyze_document_attachment",
            "analyze_image_attachment",
        }
        if action == "web_search" and self._should_answer_from_attachment(state):
            action = "final_answer"
            action_input = {}
        if action in attachment_actions and self._should_answer_from_attachment(state):
            action = "final_answer"
            action_input = {}
        if action in attachment_actions:
            action_input = self._with_attachment_args(
                action,
                action_input,
                state["question"],
            )
        if action == "run_python_attachment" and not action_input.get("file_name", "").lower().endswith(".py"):
            action = "final_answer"
            action_input = {}
        return {
            "action": action,
            "action_input": action_input,
            "draft_answer": action_input.get("answer", "").strip() if action == "final_answer" else "",
            "steps": [*state["steps"], f"decide_action:{action}"],
        }

    @traceable(name="reason")
    async def _reason(self, state: GaiaState) -> dict[str, object]:
        try:
            reasoning = await self._complete_text(
                REASONING_PROMPT,
                self._format_reasoning_context(state),
                max_tokens=self.answer_max_tokens,
                step_name="Reasoning step",
            )
        except AgentError as error:
            reasoning = self._fallback_reasoning(state, error)

        try:
            answer = await self._complete_text(
                FINAL_ANSWER_PROMPT,
                self._format_extract_context(state, reasoning),
                max_tokens=self.answer_max_tokens,
                step_name="Final answer step",
            )
        except AgentError:
            answer = self._best_effort_non_empty_answer(state)
        return {
            "reasoning": reasoning,
            "answer": self._finalize_answer(answer),
            "steps": [*state["steps"], "reason:final_answer"],
        }

    @staticmethod
    def _finalize_answer(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        answer = lines[-1] if lines else text.strip()
        return answer.strip("*`_ ").strip()

    def _fallback_reasoning(self, state: GaiaState, error: AgentError) -> str:
        successful, errors = self._split_observations(state)
        sections = [
            f"Reasoning model did not return text: {error}",
            "Use the available observations to answer if possible.",
            "Successful tool observations:",
            json.dumps(successful, ensure_ascii=False, indent=2),
        ]
        if errors:
            sections.extend([
                "Tool errors:",
                json.dumps(errors, ensure_ascii=False, indent=2),
            ])
        draft = state.get("draft_answer", "").strip()
        if draft:
            sections.extend(["Draft answer:", draft])
        return "\n\n".join(sections)

    def _best_effort_non_empty_answer(self, state: GaiaState) -> str:
        draft = state.get("draft_answer", "").strip()
        if draft:
            return draft

        successful, _errors = self._split_observations(state)
        for observation in reversed(successful):
            text = observation.get("observation", "").strip()
            if text:
                return text[:500]
        return "Unable to determine"

    def _split_observations(
        self,
        state: GaiaState,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        successful: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        for observation in state["observations"]:
            text = observation.get("observation", "")
            if self._is_tool_error(text):
                errors.append(observation)
            else:
                successful.append(observation)
        return successful, errors

    def _should_answer_from_attachment(self, state: GaiaState) -> bool:
        if self._question_requires_external_information(state["question"]):
            return False

        successful, _errors = self._split_observations(state)
        return any(
            observation.get("action") in {
                "analyze_document_attachment",
                "analyze_image_attachment",
                "run_python_attachment",
            }
            for observation in successful
        )

    @staticmethod
    def _question_requires_external_information(question: str) -> bool:
        external_markers = [
            "web",
            "website",
            "internet",
            "online",
            "search",
            "current",
            "latest",
            "today",
            "news",
            "url",
            "link",
            "page",
            "site",
        ]
        question_lower = question.lower()
        return any(marker in question_lower for marker in external_markers)

    def _format_reasoning_context(self, state: GaiaState) -> str:
        successful, errors = self._split_observations(state)
        sections = [
            f"Question:\n{state['question']}",
            "Successful tool observations:\n"
            + json.dumps(successful, ensure_ascii=False, indent=2),
            "Tool errors (secondary context only):\n"
            + json.dumps(errors, ensure_ascii=False, indent=2),
        ]
        draft = state.get("draft_answer", "").strip()
        if draft:
            sections.append(
                "Preliminary draft answer from the router (unverified, may be wrong):\n"
                + draft
            )
        sections.append(
            "Reason step by step over the observations and determine the answer to the question."
        )
        return "\n\n".join(sections)

    def _format_extract_context(self, state: GaiaState, reasoning: str) -> str:
        return (
            f"Original question:\n{state['question']}\n\n"
            f"Reasoning trace:\n{reasoning}\n\n"
            "Return only the final answer to the question, in exactly the requested format."
        )

    def _is_tool_error(self, text: str) -> bool:
        error_markers = [
            "failed:",
            "error:",
            "not supported",
            "only supports",
            "syntaxerror",
            "timed out",
            "could not",
            "no local file was found",
        ]
        lowered = text.lower()
        return any(marker in lowered for marker in error_markers)

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
            return "final_answer", {}

        normalized_input = {str(key): str(value)
                            for key, value in action_input.items()}
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
            return "reason"
        if state["step_count"] >= self.max_steps:
            return "reason"
        return "execute_tool"

    def _route_after_tool(self, state: GaiaState) -> str:
        if state["step_count"] >= self.max_steps:
            return "reason"
        return "decide_action"

    def _with_attachment_args(
        self,
        action: str,
        action_input: dict[str, str],
        question: str,
    ) -> dict[str, str]:
        # The model sometimes omits required attachment arguments; backfill them
        # from the task text so a forgetful tool call still runs.
        enriched = dict(action_input)
        if not enriched.get("task_id", "").strip():
            task_id = self._field_from_question(question, "task id")
            if task_id:
                enriched["task_id"] = task_id
        if not enriched.get("file_name", "").strip():
            file_name = self._field_from_question(question, "attached file")
            if file_name:
                enriched["file_name"] = file_name
        needs_question = action in {
            "analyze_document_attachment",
            "analyze_image_attachment",
        }
        if needs_question and not enriched.get("question", "").strip():
            enriched["question"] = self._question_body(question)
        return enriched

    @staticmethod
    def _field_from_question(question: str, label: str) -> str:
        for line in question.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == label:
                return value.strip()
        return ""

    @staticmethod
    def _question_body(question: str) -> str:
        lines = question.splitlines()
        for index, line in enumerate(lines):
            if line.strip().lower().rstrip(":") == "question":
                body = "\n".join(lines[index + 1:]).strip()
                if body:
                    return body
        return question.strip()

    async def _invoke_tool(self, action: str, tool_input: dict[str, str]) -> str:
        try:
            return await ASYNC_AGENT_TOOLS[action](**tool_input)
        except Exception as error:
            return f"{action} failed: {type(error).__name__}: {error}"

    @traceable(name="execute_tool")
    async def _execute_tool(self, state: GaiaState) -> dict[str, object]:
        action = state["action"]
        action_input = state["action_input"]
        observation = ""
        update: dict[str, object] = {
            "step_count": state["step_count"] + 1,
            "steps": [*state["steps"], f"execute_tool:{action}"],
        }

        if action == "web_search":
            query = action_input.get("query", "").strip() or state["question"]
            observation = await self._invoke_tool(action, {"query": query})
            update["search_query"] = query
            update["search_results"] = observation
        elif action == "search_url":
            url = action_input.get("url", "").strip()
            observation = await self._invoke_tool(action, {"url": url})
            update["page_url"] = url
            update["page_content"] = observation
        elif action in AGENT_TOOLS:
            if action in {
                "run_python_attachment",
                "analyze_document_attachment",
                "analyze_image_attachment",
            }:
                action_input = self._with_attachment_args(
                    action,
                    action_input,
                    state["question"],
                )
            observation = await self._invoke_tool(action, action_input)
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
