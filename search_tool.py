"""
A free, no-API-key web search tool for our CrewAI agent, built on DuckDuckGo.

Note for beginners: the old `duckduckgo-search` PyPI package was renamed to
`ddgs` in 2025. We use `ddgs` here because it's the actively maintained one.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ddgs import DDGS


class DuckDuckGoSearchInput(BaseModel):
    """Schema describing the single input this tool accepts."""
    query: str = Field(..., description="The search query to look up on the web.")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "Searches the web using DuckDuckGo and returns the top results "
        "(title, link, and short summary) for a given query. Use this "
        "tool whenever you need current, real-world information — the "
        "language model on its own does not know about recent events."
    )
    args_schema: type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
        except Exception as exc:  # keep the agent alive even if a search fails
            return f"Search failed for query '{query}': {exc}"

        if not results:
            return f"No results found for '{query}'. Try a different query."

        formatted = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No title")
            link = r.get("href", "No link")
            snippet = r.get("body", "No description")
            formatted.append(f"{i}. {title}\n   Link: {link}\n   Summary: {snippet}")

        return "\n\n".join(formatted)
