from langchain_core.tools import tool

if __package__:
    from .tool_helpers import (
        analyze_document_attachment_file,
        analyze_image_attachment_file,
        read_url_text,
        run_python_attachment_file,
        web_search_text,
    )
else:
    from tool_helpers import (
        analyze_document_attachment_file,
        analyze_image_attachment_file,
        read_url_text,
        run_python_attachment_file,
        web_search_text,
    )


@tool
def web_search(query: str) -> str:
    """Search the web for relevant sources using a concise keyword query."""
    return web_search_text(query)


@tool
def search_url(url: str) -> str:
    """Read a specific URL and extract readable page text."""
    return read_url_text(url)


@tool
def run_python_attachment(task_id: str, file_name: str = "", timeout_seconds: int = 5) -> str:
    """Download or read an attached Python file, execute it in Docker, and return the output."""
    return run_python_attachment_file(task_id, file_name, timeout_seconds)


@tool
def analyze_document_attachment(task_id: str, question: str = "", file_name: str = "") -> str:
    """Download or read an attached PDF, spreadsheet, or text file and return extracted content."""
    return analyze_document_attachment_file(task_id, question, file_name)


@tool
def analyze_image_attachment(task_id: str, question: str, file_name: str = "") -> str:
    """Download or read an attached image and return structured visual observations."""
    return analyze_image_attachment_file(task_id, question, file_name)


AGENT_TOOLS = {
    web_search.name: web_search,
    search_url.name: search_url,
    run_python_attachment.name: run_python_attachment,
    analyze_document_attachment.name: analyze_document_attachment,
    analyze_image_attachment.name: analyze_image_attachment,
}
