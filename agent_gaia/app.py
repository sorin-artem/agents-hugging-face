import os
import json

import gradio as gr
import requests
from dotenv import load_dotenv
from langsmith import traceable

from agent_langgraph import GaiaAgent


load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"
TASK_IDS = {
    "Mercedes Sosa studio albums": "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
    "Reversed text answer": "2d83110e-a098-4ebb-9987-066c06fa42d0",
    "Wikipedia dinosaur featured article": "4fc2f1ae-8625-45b5-ab34-ad4433bc21f8",
}
GRAPH_HTML = """
<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin:16px 0;">
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">START</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">decide_action</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">execute_tool</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">decide_action / final_answer</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">END</div>
</div>
<div style="font-size: 0.9em; opacity: 0.8;">
  The agent loops between decide_action and execute_tool until it returns final_answer or reaches max steps.
</div>
"""

agent = GaiaAgent()


def get_question(task_id: str) -> str:
    return get_questions_by_id()[task_id]


def get_questions_by_id() -> dict[str, str]:
    response = requests.get(f"{API_URL}/questions", timeout=30)
    response.raise_for_status()
    questions = response.json()

    questions_by_id = {
        item.get("task_id"): item.get("question", "")
        for item in questions
    }
    missing_ids = [
        task_id for task_id in TASK_IDS.values()
        if task_id not in questions_by_id
    ]
    if missing_ids:
        raise ValueError(f"Questions not found: {', '.join(missing_ids)}")

    return questions_by_id


def format_questions_markdown() -> str:
    questions_by_id = get_questions_by_id()
    lines = ["## Selected Benchmark Questions"]
    for index, (label, task_id) in enumerate(TASK_IDS.items(), start=1):
        question = questions_by_id[task_id]
        lines.append(f"{index}. **{label}**  \n`{task_id}`  \n{question}")
    return "\n\n".join(lines)


def run_question_trace(task_label: str) -> tuple[str, str, str, str, str, str, str, str]:
    task_id = TASK_IDS[task_label]
    question = get_question(task_id)
    result = agent.run(question)

    steps = " -> ".join(result["steps"])
    return (
        task_id,
        question,
        steps,
        result["search_query"],
        result["search_results"],
        result["page_url"],
        result["page_content"],
        result["answer"],
    )


def build_agent_code_url(agent_code: str) -> str:
    if agent_code.strip():
        return agent_code.strip()

    space_id = os.getenv("SPACE_ID")
    if space_id:
        return f"https://huggingface.co/spaces/{space_id}/tree/main"

    return "local-development"


@traceable(name="run_single_question")
def run_single_question(label: str, task_id: str, question: str) -> dict[str, object]:
    result = agent.run(question)
    return {
        "label": label,
        "task_id": task_id,
        "question": question,
        "answer": result["answer"].strip(),
        "steps": result["steps"],
        "search_query": result["search_query"],
        "search_results": result["search_results"],
        "page_url": result["page_url"],
        "page_content": result["page_content"],
    }


@traceable(name="run_all_questions")
def run_all_questions(username: str, agent_code: str) -> dict[str, object]:
    questions_by_id = get_questions_by_id()
    runs = [
        run_single_question(label, task_id, questions_by_id[task_id])
        for label, task_id in TASK_IDS.items()
    ]
    answers_payload = [
        {
            "task_id": str(run["task_id"]),
            "submitted_answer": str(run["answer"]),
        }
        for run in runs
    ]

    payload = {
        "username": username.strip(),
        "agent_code": build_agent_code_url(agent_code),
        "answers": answers_payload,
    }

    response = requests.post(f"{API_URL}/submit", json=payload, timeout=60)
    response.raise_for_status()
    submission_result = response.json()
    status = (
        f"Score: {submission_result.get('score', 'N/A')}% "
        f"({submission_result.get('correct_count', '?')}/"
        f"{submission_result.get('total_attempted', '?')} correct)\n"
        f"Message: {submission_result.get('message', 'No message')}"
    )

    return {
        "status": status,
        "submission_result": submission_result,
        "answers": answers_payload,
        "runs": runs,
    }


def run_and_submit_all(username: str, agent_code: str) -> tuple[str, str]:
    if not username.strip():
        return "Enter your Hugging Face username.", ""

    result = run_all_questions(username, agent_code)
    return (
        str(result["status"]),
        json.dumps(result["runs"], indent=2, ensure_ascii=False),
    )


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent")
    gr.Markdown("## Current LangGraph Workflow")
    gr.HTML(GRAPH_HTML)
    questions_markdown = gr.Markdown()
    task_selector = gr.Dropdown(
        label="Benchmark question",
        choices=list(TASK_IDS.keys()),
        value=next(iter(TASK_IDS)),
    )
    trace_button = gr.Button("Run selected question with trace")
    trace_task_id_output = gr.Textbox(label="Task ID")
    trace_question_output = gr.Textbox(label="Question", lines=5)
    trace_steps_output = gr.Textbox(label="Executed steps")
    trace_query_output = gr.Textbox(label="Search query")
    trace_results_output = gr.Textbox(label="Search results", lines=10)
    trace_url_output = gr.Textbox(label="Fetched URL")
    trace_page_output = gr.Textbox(label="Fetched page content", lines=10)
    trace_answer_output = gr.Textbox(label="Final answer")
    trace_button.click(
        fn=run_question_trace,
        inputs=task_selector,
        outputs=[
            trace_task_id_output,
            trace_question_output,
            trace_steps_output,
            trace_query_output,
            trace_results_output,
            trace_url_output,
            trace_page_output,
            trace_answer_output,
        ],
    )

    gr.Markdown("## Check All 3 Answers")
    username_input = gr.Textbox(
        label="Hugging Face username",
        value=os.getenv("HF_USERNAME", "sorin-artem"),
    )
    agent_code_input = gr.Textbox(
        label="Agent code URL",
        placeholder="Optional locally. In Space, SPACE_ID is used automatically.",
    )
    submit_button = gr.Button("Run all 3 and submit for checking")
    submit_status_output = gr.Textbox(label="Check result", lines=3)
    submit_log_output = gr.Code(label="Run log", language="json", lines=16)
    submit_button.click(
        fn=run_and_submit_all,
        inputs=[
            username_input,
            agent_code_input,
        ],
        outputs=[submit_status_output, submit_log_output],
    )
    demo.load(fn=format_questions_markdown,
              inputs=None, outputs=questions_markdown)


if __name__ == "__main__":
    demo.launch()
