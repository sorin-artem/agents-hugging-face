from datetime import datetime
import os

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "Local Demo MCP Server",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8000")),
    sse_path=os.getenv("MCP_SSE_PATH", "/sse"),
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def echo(message: str) -> str:
    """Return the provided message."""
    return f"Echo: {message}"


@mcp.tool()
def current_time() -> str:
    """Return the current local time in ISO 8601 format."""
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
