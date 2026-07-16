import os
import json
import asyncio
import time

import gradio as gr
import httpx
from dotenv import load_dotenv
from langsmith import traceable

from agent_langgraph import GaiaAgent


load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"
SUPPORTED_FILE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".py",
    ".pdf",
    ".xlsx",
    ".xlsm",
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".json",
    ".xml",
}
UNSUPPORTED_FILE_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
}
DEFAULT_FILE_TASK_IDS = [
    "cca530fc-4052-43b2-b130-b30968d8aa44",  # chess position image
    "f918266a-b3e0-4914-865d-4faa564f1aef",  # Python code file
    "7bd855d8-463d-4ed5-93ca-5fe35145f733",  # Excel spreadsheet
    # "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3",  # recipe audio
    # "1f975693-876d-457b-a649-393859e79bf3",  # homework audio
]
CHECK_QUESTION_LIMIT = int(
    os.getenv("CHECK_QUESTION_LIMIT", "20"))
CHECK_MODEL = os.getenv("CHECK_OPENROUTER_MODEL", "z-ai/glm-5.2")
CHECK_QUESTION_MODE = os.getenv("CHECK_QUESTION_MODE", "all").strip().lower()
CHECK_QUESTION_TIMEOUT_SECONDS = int(
    os.getenv("CHECK_QUESTION_TIMEOUT_SECONDS", "300"))


def optional_int_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"", "0", "none", "unlimited", "-1"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# None here means "unset" — the agent then falls back to its own generous default
# ceiling. It does NOT mean unlimited output (the provider truncates long reasoning
# when no cap is sent). Set the env to a number to override, or "none" to opt out.
CHECK_ACTION_MAX_TOKENS = optional_int_env("CHECK_OPENROUTER_ACTION_MAX_TOKENS")
CHECK_ANSWER_MAX_TOKENS = optional_int_env("CHECK_OPENROUTER_ANSWER_MAX_TOKENS")
CHECK_MAX_STEPS = int(os.getenv("CHECK_AGENT_MAX_STEPS", "6"))


async def get_questions() -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{API_URL}/questions")
        response.raise_for_status()
        return response.json()


def file_extension(file_name: str) -> str:
    return os.path.splitext(file_name.strip())[1].lower()


def is_supported_file_question(item: dict[str, str]) -> bool:
    file_name = str(item.get("file_name", ""))
    return bool(file_name) and file_extension(file_name) in SUPPORTED_FILE_EXTENSIONS


def sort_file_questions(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred_order = {
        task_id: index
        for index, task_id in enumerate(DEFAULT_FILE_TASK_IDS)
    }
    return sorted(
        questions,
        key=lambda item: (
            preferred_order.get(str(item.get("task_id", "")), len(preferred_order)),
            str(item.get("file_name", "")),
        ),
    )


async def get_supported_file_questions() -> list[dict[str, str]]:
    questions = [
        item
        for item in await get_questions()
        if item.get("task_id") and item.get("question") and is_supported_file_question(item)
    ]
    return sort_file_questions(questions)


async def get_valid_questions() -> list[dict[str, str]]:
    return [
        item
        for item in await get_questions()
        if item.get("task_id") and item.get("question")
    ]


def is_audio_file_question(item: dict[str, str]) -> bool:
    file_name = str(item.get("file_name", ""))
    return bool(file_name) and file_extension(file_name) in UNSUPPORTED_FILE_EXTENSIONS


async def get_check_questions() -> list[dict[str, str]]:
    questions = await get_valid_questions()
    questions_by_id = {str(item["task_id"]): item for item in questions}
    check_task_ids = os.getenv("CHECK_TASK_IDS", "").strip()
    if check_task_ids:
        preferred_task_ids = [
            task_id.strip()
            for task_id in check_task_ids.split(",")
            if task_id.strip()
        ]
        preferred_questions = [
            questions_by_id[task_id]
            for task_id in preferred_task_ids
            if task_id in questions_by_id
        ]
        return preferred_questions[:CHECK_QUESTION_LIMIT]

    return questions[:CHECK_QUESTION_LIMIT]


async def summarize_file_questions() -> str:
    supported = await get_supported_file_questions()
    all_file_questions = [
        item
        for item in await get_questions()
        if item.get("task_id") and item.get("question") and item.get("file_name")
    ]
    unsupported = [
        item
        for item in all_file_questions
        if file_extension(str(item.get("file_name", ""))) in UNSUPPORTED_FILE_EXTENSIONS
    ]
    selected = supported[:CHECK_QUESTION_LIMIT]
    lines = [
        f"Selected supported file tasks: {len(selected)}/{len(supported)}",
        f"Skipped unsupported audio tasks: {len(unsupported)}",
        "",
    ]
    for index, item in enumerate(selected, start=1):
        question = str(item.get("question", "")).replace("\n", " ")
        lines.append(
            f"{index}. {item.get('task_id')} | {item.get('file_name')} | {question[:140]}"
        )
    if unsupported:
        lines.extend(["", "Unsupported audio tasks:"])
        for item in unsupported:
            lines.append(f"- {item.get('task_id')} | {item.get('file_name')}")
    return "\n".join(lines)


async def summarize_check_questions() -> str:
    selected = await get_check_questions()
    lines = [
        f"Question mode: {CHECK_QUESTION_MODE}",
        f"Selected questions: {len(selected)}",
        f"Question limit: {CHECK_QUESTION_LIMIT}",
        "Parallel workers: unlimited async tasks",
        "",
    ]
    for index, item in enumerate(selected, start=1):
        question = str(item.get("question", "")).replace("\n", " ")
        file_name = str(item.get("file_name", ""))
        suffix = f" | {file_name}" if file_name else ""
        lines.append(
            f"{index}. {item.get('task_id')}{suffix} | {question[:140]}"
        )
    return "\n".join(lines)


async def preview_file_tasks() -> tuple[str, str]:
    return "Question preview loaded.", await summarize_check_questions()


def question_label(item: dict[str, str], index: int) -> str:
    task_id = str(item.get("task_id", ""))
    return str(item.get("label") or item.get("name") or f"Question {index}: {task_id}")


def build_agent_question(task_id: str, question: str, file_name: str = "") -> str:
    metadata = [f"Task ID: {task_id}"]
    if file_name:
        metadata.append(f"Attached file: {file_name}")
        metadata.append(
            "Use the relevant file tool when the answer depends on this attachment.")

    return "\n".join([*metadata, "", "Question:", question])


def build_agent_code_url(agent_code: str) -> str:
    if agent_code.strip():
        return agent_code.strip()

    space_id = os.getenv("SPACE_ID")
    if space_id:
        return f"https://huggingface.co/spaces/{space_id}/tree/main"

    return "local-development"


def create_check_agent() -> GaiaAgent:
    return GaiaAgent(
        model=CHECK_MODEL,
        action_max_tokens=CHECK_ACTION_MAX_TOKENS,
        answer_max_tokens=CHECK_ANSWER_MAX_TOKENS,
        max_steps=CHECK_MAX_STEPS,
    )


@traceable(name="run_single_question")
async def run_single_question(
    label: str,
    task_id: str,
    question: str,
    file_name: str = "",
    agent: GaiaAgent | None = None,
) -> dict[str, object]:
    agent_question = build_agent_question(task_id, question, file_name)
    result = await (agent or create_check_agent()).arun(agent_question)
    return {
        "label": label,
        "task_id": task_id,
        "file_name": file_name,
        "question": question,
        "answer": result["answer"].strip(),
        "reasoning": result.get("reasoning", ""),
        "steps": result["steps"],
        "observations": result["observations"],
        "search_query": result["search_query"],
        "search_results": result["search_results"],
        "page_url": result["page_url"],
        "page_content": result["page_content"],
    }


def failed_question_run(
    label: str,
    task_id: str,
    question: str,
    error: Exception,
    file_name: str = "",
) -> dict[str, object]:
    return {
        "label": label,
        "task_id": task_id,
        "file_name": file_name,
        "question": question,
        "answer": "",
        "reasoning": "",
        "steps": [f"error:{type(error).__name__}: {error}"],
        "observations": [],
        "search_query": "",
        "search_results": "",
        "page_url": "",
        "page_content": "",
    }


async def run_check_question(index: int, item: dict[str, str]) -> dict[str, object]:
    started_at = time.monotonic()
    label = question_label(item, index)
    task_id = str(item["task_id"])
    question = str(item["question"])
    file_name = str(item.get("file_name", ""))
    try:
        run = await run_single_question(label, task_id, question, file_name)
        run["status"] = "completed"
        run["duration_seconds"] = round(time.monotonic() - started_at, 2)
        return run
    except Exception as error:
        run = failed_question_run(label, task_id, question, error, file_name)
        run["status"] = "failed"
        run["duration_seconds"] = round(time.monotonic() - started_at, 2)
        return run


@traceable(name="run_selected_questions_parallel")
async def run_selected_questions_parallel() -> dict[str, object]:
    username = os.getenv("HF_USERNAME", "sorin-artem").strip()
    agent_code = build_agent_code_url(os.getenv("AGENT_CODE_URL", ""))
    questions = await get_check_questions()
    indexed_questions = list(enumerate(questions, start=1))

    async def run_with_limit(index: int, item: dict[str, str]) -> dict[str, object]:
        try:
            return await asyncio.wait_for(
                run_check_question(index, item),
                timeout=CHECK_QUESTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            task_id = str(item["task_id"])
            question = str(item["question"])
            file_name = str(item.get("file_name", ""))
            run = failed_question_run(
                question_label(item, index),
                task_id,
                question,
                error,
                file_name,
            )
            run["status"] = "timeout"
            run["duration_seconds"] = CHECK_QUESTION_TIMEOUT_SECONDS
            return run

    runs = await asyncio.gather(
        *[
            run_with_limit(index, item)
            for index, item in indexed_questions
        ]
    )

    answers_payload = [
        {
            "task_id": str(run["task_id"]),
            "submitted_answer": str(run["answer"]),
        }
        for run in runs
    ]
    payload = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{API_URL}/submit", json=payload)
            response.raise_for_status()
            submission_result = response.json()
    except Exception as error:
        submission_result = {
            "score": "N/A",
            "correct_count": "?",
            "total_attempted": len(answers_payload),
            "message": f"Submission failed: {type(error).__name__}: {error}",
        }
    return {
        "submission_result": submission_result,
        "answers": answers_payload,
        "runs": runs,
        "settings": {
            "model": CHECK_MODEL,
            "question_limit": CHECK_QUESTION_LIMIT,
            "concurrency": "unlimited_async_tasks",
            "question_timeout_seconds": CHECK_QUESTION_TIMEOUT_SECONDS,
            "action_max_tokens": CHECK_ACTION_MAX_TOKENS,
            "answer_max_tokens": CHECK_ANSWER_MAX_TOKENS,
            "max_steps": CHECK_MAX_STEPS,
        },
    }


async def run_and_check_selected_questions() -> tuple[str, str]:
    result = await run_selected_questions_parallel()
    submission_result = result["submission_result"]
    status = (
        f"Score: {submission_result.get('score', 'N/A')}% "
        f"({submission_result.get('correct_count', '?')}/"
        f"{submission_result.get('total_attempted', '?')} correct)\n"
        f"Message: {submission_result.get('message', 'No message')}"
    )
    return status, json.dumps(result, indent=2, ensure_ascii=False)


with gr.Blocks() as demo:
    gr.Markdown("# General Tool Agent")
    gr.Markdown(
        "Runs selected questions in parallel, then submits the answers to the "
        "Hugging Face scoring API. Configure CHECK_QUESTION_MODE=all, no_audio, "
        "or file_supported."
    )
    preview_button = gr.Button("Preview selected questions")
    submit_button = gr.Button("Run selected questions and check")
    submit_status_output = gr.Textbox(label="Check result", lines=3)
    submit_log_output = gr.Code(label="Run log", language="json", lines=20)
    preview_button.click(
        fn=preview_file_tasks,
        inputs=None,
        outputs=[submit_status_output, submit_log_output],
    )
    submit_button.click(
        fn=run_and_check_selected_questions,
        inputs=None,
        outputs=[submit_status_output, submit_log_output],
    )


if __name__ == "__main__":
    demo.launch()
