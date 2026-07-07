from __future__ import annotations

from ddgs import DDGS
from langchain_core.tools import tool
from trafilatura import extract, fetch_url


MAX_PAGE_CHARS = 12000


@tool
def web_search(query: str) -> str:
    """Search the web for relevant sources using a concise keyword query."""
    query = query.strip()
    if not query:
        return "No search query provided."

    results = DDGS().text(query, max_results=8)

    if not results:
        return "No search results found."

    formatted_results = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("href", "")
        snippet = result.get("body", "")
        formatted_results.append(f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")

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


AGENT_TOOLS = {
    web_search.name: web_search,
    search_url.name: search_url,
}
