from __future__ import annotations

from ddgs import DDGS
from trafilatura import extract, fetch_url


def web_search(query: str, max_results: int = 5) -> str:
    query = query.strip()
    if not query:
        return "No search query provided."

    results = DDGS().text(query, max_results=max_results)

    if not results:
        return "No search results found."

    formatted_results = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("href", "")
        snippet = result.get("body", "")
        formatted_results.append(f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")

    return "\n\n".join(formatted_results)


def search_url(url: str, max_chars: int = 12000) -> str:
    url = url.strip()
    if not url:
        return "No URL provided."

    downloaded = fetch_url(url)
    if not downloaded:
        return f"Could not fetch URL: {url}"

    text = extract(downloaded, url=url, include_comments=False)
    if not text:
        return f"Could not extract readable text from URL: {url}"

    return text[:max_chars]
