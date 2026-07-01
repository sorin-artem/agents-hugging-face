import os

import gradio as gr
import requests
from dotenv import load_dotenv

from agent_langgraph import GaiaAgent


load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"
TASK_ID = "8e867cd7-cff9-4e6c-867a-ff5ddc2550be"
GRAPH_HTML = """
<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin:16px 0;">
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">START</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">create_search_query</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">web_search</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">choose_url</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">fetch_url</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">answer</div>
  <div>→</div>
  <div style="padding:12px 16px; border:1px solid #888; border-radius:8px;">END</div>
</div>
"""

agent = GaiaAgent()


def get_fixed_question() -> str:
    response = requests.get(f"{API_URL}/questions", timeout=30)
    response.raise_for_status()
    questions = response.json()

    question_data = next(
        (item for item in questions if item.get("task_id") == TASK_ID), None)
    if question_data is None:
        raise ValueError(f"Question not found: {TASK_ID}")

    return question_data.get("question", "")


def run_fixed_question_trace() -> tuple[str, str, str, str, str, str, str, str]:
    question = get_fixed_question()
    result = agent.run(question)

    steps = " -> ".join(result["steps"])
    return (
        TASK_ID,
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


def submit_answer(username: str, agent_code: str, answer: str) -> str:
    if not username.strip():
        return "Enter your Hugging Face username."

    if not answer.strip():
        return "Run the agent first or enter a final answer."

    payload = {
        "username": username.strip(),
        "agent_code": build_agent_code_url(agent_code),
        "answers": [
            {
                "task_id": TASK_ID,
                "submitted_answer": answer.strip(),
            }
        ],
    }

    response = requests.post(f"{API_URL}/submit", json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    return (
        f"Score: {result.get('score', 'N/A')}% "
        f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)\n"
        f"Message: {result.get('message', 'No message')}"
    )


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent")
    gr.Markdown("## Current LangGraph Workflow")
    gr.HTML(GRAPH_HTML)
    trace_button = gr.Button("Run fixed question with trace")
    trace_task_id_output = gr.Textbox(label="Task ID")
    trace_question_output = gr.Textbox(label="Question", lines=5)
    trace_steps_output = gr.Textbox(label="Executed steps")
    trace_query_output = gr.Textbox(label="Search query")
    trace_results_output = gr.Textbox(label="Search results", lines=10)
    trace_url_output = gr.Textbox(label="Fetched URL")
    trace_page_output = gr.Textbox(label="Fetched page content", lines=10)
    trace_answer_output = gr.Textbox(label="Final answer")
    trace_button.click(
        fn=run_fixed_question_trace,
        inputs=None,
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

    gr.Markdown("## Check Answer")
    username_input = gr.Textbox(
        label="Hugging Face username",
        value=os.getenv("HF_USERNAME", ""),
    )
    agent_code_input = gr.Textbox(
        label="Agent code URL",
        placeholder="Optional locally. In Space, SPACE_ID is used automatically.",
    )
    submit_button = gr.Button("Submit answer for checking")
    submit_status_output = gr.Textbox(label="Check result", lines=3)
    submit_button.click(
        fn=submit_answer,
        inputs=[username_input, agent_code_input, trace_answer_output],
        outputs=submit_status_output,
    )


if __name__ == "__main__":
    demo.launch()
