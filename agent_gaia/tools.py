from __future__ import annotations

import base64
import os
from pathlib import Path

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import tool
from openai import OpenAI
import requests
from trafilatura import extract, fetch_url


TASK_FILE_API_URL = os.getenv(
    "TASK_FILE_API_URL",
    os.getenv("GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"),
)
MAX_PAGE_CHARS = 12000
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _guess_image_mime(content: bytes, content_type: str) -> str:
    content_type = content_type.split(";")[0].strip().lower()
    if content_type.startswith("image/"):
        return content_type
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
        return "image/gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _vision_model() -> str:
    return os.getenv(
        "OPENROUTER_VISION_MODEL",
        "google/gemini-2.5-flash",
    )


def _local_file_candidates(task_id: str, file_name: str) -> list[Path]:
    base_dirs = [
        Path(os.getenv("ATTACHMENT_FILES_DIR", os.getenv("GAIA_FILES_DIR", ""))),
        Path.cwd(),
        Path.cwd() / "files",
        Path.cwd() / "downloads",
        Path.cwd() / "agent_gaia" / "files",
    ]
    names = [file_name] if file_name else []
    names.extend([f"{task_id}{suffix}" for suffix in [".png", ".jpg", ".jpeg", ".webp", ".gif"]])
    return [
        directory / name
        for directory in base_dirs
        if str(directory) != "."
        for name in names
        if name
    ]


def _download_or_read_task_file(task_id: str, file_name: str) -> tuple[bytes, str, str]:
    try:
        response = requests.get(f"{TASK_FILE_API_URL}/files/{task_id}", timeout=60)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", ""), "task file API"
    except requests.RequestException as api_error:
        for candidate in _local_file_candidates(task_id, file_name):
            if candidate.is_file():
                return candidate.read_bytes(), "", str(candidate)
        raise RuntimeError(
            f"Could not download file for task_id {task_id} and no local file was found. "
            f"Last API error: {api_error}"
        ) from api_error


@tool
def web_search(query: str) -> str:
    """Search the web for relevant sources using a concise keyword query."""
    query = query.strip()
    if not query:
        return "No search query provided."

    try:
        results = DDGS().text(query, max_results=8)
    except DDGSException as error:
        return f"Search failed: {error}"

    if not results:
        return "No search results found."

    formatted_results = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("href", "")
        snippet = result.get("body", "")
        formatted_results.append(
            f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")

    return "\n\n".join(formatted_results)


@tool
def search_url(url: str) -> str:
    """Read a specific URL and extract readable page text."""
    url = url.strip()
    if not url:
        return "No URL provided."

    downloaded = fetch_url(url)
    if not downloaded:
        return f"Could not fetch URL: {url}"

    text = extract(downloaded, url=url, include_comments=False)
    if not text:
        return f"Could not extract readable text from URL: {url}"

    return text[:MAX_PAGE_CHARS]


@tool
def analyze_image_attachment(task_id: str, question: str, file_name: str = "") -> str:
    """Download or read an attached image and return structured visual observations."""
    task_id = task_id.strip()
    question = question.strip()
    file_name = file_name.strip()
    if not task_id:
        return "No task_id provided."
    if not question:
        return "No image question provided."

    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        return "Set OPEN_ROUTER_API_KEY to use image analysis."

    try:
        image_bytes, content_type, source = _download_or_read_task_file(task_id, file_name)
    except RuntimeError as error:
        return str(error)

    if not image_bytes:
        return f"Loaded file for task_id {task_id}, but it was empty."
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return f"Image is too large to analyze ({len(image_bytes)} bytes)."

    mime_type = _guess_image_mime(image_bytes, content_type)
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        completion = client.chat.completions.create(
            model=_vision_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a visual analysis tool. Do not solve the user's task directly "
                        "unless it only asks for a plain description. Return structured observations "
                        "that another agent can use to answer. Be careful about visible text, labels, "
                        "coordinates, orientation, counts, spatial relationships, and uncertainty. "
                        "If the image is a board, chart, table, diagram, map, or puzzle, describe the "
                        "layout and all visible elements precisely instead of guessing the final answer. "
                        "Use plain text only, no markdown formatting."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "User question:\n"
                                f"{question}\n\n"
                                "Return this format:\n"
                                "Visual analysis:\n"
                                "Image type: ...\n"
                                "Relevant visible text/labels: ...\n"
                                "Key objects/entities and positions: ...\n"
                                "Spatial relationships/orientation: ...\n"
                                "Details relevant to the user's question: ...\n"
                                "Uncertainties: ...\n"
                                "Do not include a final answer line unless the user only asked for image description."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=1024,
        )
    except Exception as error:
        return f"Image analysis failed: {type(error).__name__}: {error}"

    analysis = completion.choices[0].message.content or ""
    return f"Image source: {source}\n{analysis.strip()}"


AGENT_TOOLS = {
    web_search.name: web_search,
    search_url.name: search_url,
    analyze_image_attachment.name: analyze_image_attachment,
}
