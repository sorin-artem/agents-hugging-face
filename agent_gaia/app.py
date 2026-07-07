import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import requests
from dotenv import load_dotenv
from langsmith import traceable

from agent_langgraph import GaiaAgent


load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_FILE_TASK_IDS = [
    "cca530fc-4052-43b2-b130-b30968d8aa44",  # chess position image
    # "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3",  # recipe audio
    # "f918266a-b3e0-4914-865d-4faa564f1aef",  # Python code file
    # "1f975693-876d-457b-a649-393859e79bf3",  # homework audio
    # "7bd855d8-463d-4ed5-93ca-5fe35145f733",  # Excel spreadsheet
]
CHECK_QUESTION_LIMIT = int(
    os.getenv("CHECK_QUESTION_LIMIT", str(len(DEFAULT_FILE_TASK_IDS))))
CHECK_MAX_WORKERS = int(
    os.getenv("CHECK_MAX_WORKERS", str(CHECK_QUESTION_LIMIT)))
CHECK_MODEL = os.getenv("CHECK_OPENROUTER_MODEL", "z-ai/glm-5.2")
CHECK_ACTION_MAX_TOKENS = int(
    os.getenv("CHECK_OPENROUTER_ACTION_MAX_TOKENS", "512"))
CHECK_ANSWER_MAX_TOKENS = int(
    os.getenv("CHECK_OPENROUTER_ANSWER_MAX_TOKENS", "512"))
CHECK_MAX_STEPS = int(os.getenv("CHECK_AGENT_MAX_STEPS", "6"))


def get_questions() -> list[dict[str, str]]:
    response = requests.get(f"{API_URL}/questions", timeout=30)
    response.raise_for_status()
    return response.json()


def get_check_questions() -> list[dict[str, str]]:
    questions = [
        item
        for item in get_questions()
        if item.get("task_id") and item.get("question")
    ]
    questions_by_id = {str(item["task_id"]): item for item in questions}
    preferred_task_ids = [
        task_id.strip()
        for task_id in os.getenv("CHECK_TASK_IDS", ",".join(DEFAULT_FILE_TASK_IDS)).split(",")
        if task_id.strip()
    ]
    preferred_questions = [
        questions_by_id[task_id]
        for task_id in preferred_task_ids
        if task_id in questions_by_id
    ]
    return preferred_questions[:CHECK_QUESTION_LIMIT]


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
def run_single_question(
    label: str,
    task_id: str,
    question: str,
    file_name: str = "",
    agent: GaiaAgent | None = None,
) -> dict[str, object]:
    agent_question = build_agent_question(task_id, question, file_name)
    result = (agent or create_check_agent()).run(agent_question)
    return {
        "label": label,
        "task_id": task_id,
        "file_name": file_name,
        "question": question,
        "answer": result["answer"].strip(),
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
) -> dict[str, object]:
    return {
        "label": label,
        "task_id": task_id,
        "question": question,
        "answer": "",
        "steps": [f"error:{type(error).__name__}: {error}"],
        "observations": [],
        "search_query": "",
        "search_results": "",
        "page_url": "",
        "page_content": "",
    }


def run_check_question(index: int, item: dict[str, str]) -> dict[str, object]:
    label = question_label(item, index)
    task_id = str(item["task_id"])
    question = str(item["question"])
    file_name = str(item.get("file_name", ""))
    try:
        return run_single_question(label, task_id, question, file_name)
    except Exception as error:
        return failed_question_run(label, task_id, question, error)


@traceable(name="run_five_questions_parallel")
def run_five_questions_parallel() -> dict[str, object]:
    username = os.getenv("HF_USERNAME", "sorin-artem").strip()
    agent_code = build_agent_code_url(os.getenv("AGENT_CODE_URL", ""))
    questions = get_check_questions()
    indexed_questions = list(enumerate(questions, start=1))
    runs_by_index: dict[int, dict[str, object]] = {}

    with ThreadPoolExecutor(max_workers=CHECK_MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_check_question, index, item): index
            for index, item in indexed_questions
        }
        for future in as_completed(futures):
            runs_by_index[futures[future]] = future.result()

    runs = [
        runs_by_index[index]
        for index, _item in indexed_questions
        if index in runs_by_index
    ]
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

    response = requests.post(f"{API_URL}/submit", json=payload, timeout=60)
    response.raise_for_status()
    submission_result = response.json()
    return {
        "submission_result": submission_result,
        "answers": answers_payload,
        "runs": runs,
        "settings": {
            "model": CHECK_MODEL,
            "question_limit": CHECK_QUESTION_LIMIT,
            "max_workers": CHECK_MAX_WORKERS,
            "action_max_tokens": CHECK_ACTION_MAX_TOKENS,
            "answer_max_tokens": CHECK_ANSWER_MAX_TOKENS,
            "max_steps": CHECK_MAX_STEPS,
        },
    }


def run_and_check_five_questions() -> tuple[str, str]:
    result = run_five_questions_parallel()
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
        "Hugging Face scoring API."
    )
    submit_button = gr.Button("Run selected questions and check")
    submit_status_output = gr.Textbox(label="Check result", lines=3)
    submit_log_output = gr.Code(label="Run log", language="json", lines=20)
    submit_button.click(
        fn=run_and_check_five_questions,
        inputs=None,
        outputs=[submit_status_output, submit_log_output],
    )


if __name__ == "__main__":
    demo.launch()
