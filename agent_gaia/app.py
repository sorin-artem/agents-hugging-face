import os
import asyncio
import time

import gradio as gr
import pandas as pd
import requests
import spaces
from dotenv import load_dotenv

from agent_langgraph import GaiaAgent


load_dotenv()

# ZeroGPU Spaces require at least one @spaces.GPU function at startup.
# The agent itself uses OpenRouter (no local GPU), so this is only a platform hook.
@spaces.GPU(duration=60)
def _zero_gpu_placeholder() -> str:
    return "ok"


# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
QUESTION_TIMEOUT_SECONDS = int(os.getenv("CHECK_QUESTION_TIMEOUT_SECONDS", "300"))
CHECK_MODEL = os.getenv("CHECK_OPENROUTER_MODEL", os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2"))
CHECK_MAX_STEPS = int(os.getenv("CHECK_AGENT_MAX_STEPS", os.getenv("AGENT_MAX_STEPS", "8")))


def optional_int_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"", "0", "none", "unlimited", "-1"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


CHECK_ACTION_MAX_TOKENS = optional_int_env("CHECK_OPENROUTER_ACTION_MAX_TOKENS")
CHECK_ANSWER_MAX_TOKENS = optional_int_env("CHECK_OPENROUTER_ANSWER_MAX_TOKENS")


def build_agent_question(task_id: str, question: str, file_name: str = "") -> str:
    metadata = [f"Task ID: {task_id}"]
    if file_name:
        metadata.append(f"Attached file: {file_name}")
        metadata.append(
            "Use the relevant file tool when the answer depends on this attachment."
        )
    return "\n".join([*metadata, "", "Question:", question])


def create_agent() -> GaiaAgent:
    return GaiaAgent(
        model=CHECK_MODEL,
        action_max_tokens=CHECK_ACTION_MAX_TOKENS,
        answer_max_tokens=CHECK_ANSWER_MAX_TOKENS,
        max_steps=CHECK_MAX_STEPS,
    )


async def run_one_question(agent: GaiaAgent, item: dict) -> dict:
    task_id = item.get("task_id")
    question_text = item.get("question")
    file_name = str(item.get("file_name") or "")
    started_at = time.monotonic()
    try:
        agent_question = build_agent_question(str(task_id), str(question_text), file_name)
        result = await asyncio.wait_for(
            agent.arun(agent_question),
            timeout=QUESTION_TIMEOUT_SECONDS,
        )
        answer = str(result.get("answer", "")).strip()
        return {
            "Task ID": task_id,
            "Question": question_text,
            "Submitted Answer": answer,
            "Status": "ok",
            "Duration (s)": round(time.monotonic() - started_at, 2),
        }
    except Exception as error:
        return {
            "Task ID": task_id,
            "Question": question_text,
            "Submitted Answer": f"AGENT ERROR: {type(error).__name__}: {error}",
            "Status": "error",
            "Duration (s)": round(time.monotonic() - started_at, 2),
        }


async def run_all_questions(agent: GaiaAgent, questions_data: list[dict]) -> list[dict]:
    return await asyncio.gather(
        *[run_one_question(agent, item) for item in questions_data]
    )


def run_async(coro):
    """Run async work from a sync Gradio callback, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GaiaAgent on them, submits all answers,
    and displays the results — same flow as Final_Assignment_Template.
    """
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        # Local fallback when OAuth is unavailable outside HF Spaces.
        username = os.getenv("HF_USERNAME", "").strip()
        if not username:
            print("User not logged in.")
            return "Please Login to Hugging Face with the button.", None
        print(f"User not logged in via OAuth; using HF_USERNAME={username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = create_agent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    if space_id:
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    else:
        agent_code = os.getenv(
            "AGENT_CODE_URL",
            "https://huggingface.co/spaces/agents-course/Final_Assignment_Template/tree/main",
        )
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    print(f"Running agent on {len(questions_data)} questions in parallel...")
    results_log = run_async(run_all_questions(agent, questions_data))

    answers_payload = []
    for row in results_log:
        task_id = row.get("Task ID")
        submitted_answer = row.get("Submitted Answer", "")
        if not task_id:
            continue
        if str(submitted_answer).startswith("AGENT ERROR:"):
            submitted_answer = ""
        answers_payload.append(
            {"task_id": task_id, "submitted_answer": submitted_answer}
        )

    results_df = pd.DataFrame(results_log)
    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", results_df

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = (
        f"Agent finished. Submitting {len(answers_payload)} answers "
        f"for user '{username}'..."
    )
    print(status_update)

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/"
            f"{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        return status_message, results_df


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
**Instructions:**

1. Deploy this as a Hugging Face Space (or run locally), with `OPEN_ROUTER_API_KEY` set.
2. Log in to your Hugging Face account using the button below. This uses your HF username for submission.
3. Click **Run Evaluation & Submit All Answers** to fetch questions, run the agent, submit answers, and see the score.

---
**Disclaimers:**
Once you click submit, it can take quite some time (the agent goes through all questions in parallel).
Keep your Space public so the leaderboard can link to your code.
"""
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️ SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️ SPACE_ID environment variable not found (running locally?). "
            "Repo URL cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface for GAIA Agent Evaluation...")
    # SSR is experimental on Spaces and can crash the Node side after startup.
    demo.launch(debug=True, share=False, ssr_mode=False)
