"""
Builds our single-agent CrewAI "crew".

One agent, one task:
  - The agent searches the web (DuckDuckGo) for the requested topic.
  - The agent writes up a structured Markdown report on what it found.

Kept deliberately simple/beginner-friendly: no multi-agent hierarchy,
no delegation, no memory store.
"""

import litellm

from crewai import Agent, Task, Crew, Process, LLM
from tools.search_tool import DuckDuckGoSearchTool


# ---------------------------------------------------------------------------
# Workaround for a known CrewAI bug (as of crewai 1.15.22, tracked at
# https://github.com/crewAIInc/crewAI/issues/6789 — fix not yet released).
#
# CrewAI tags messages with an internal "cache_breakpoint" field meant only
# for Anthropic's prompt-caching feature, but fails to strip that field
# before sending requests through LiteLLM to other providers. Groq's API
# validates strictly and rejects the unknown field with a BadRequestError.
#
# This patches litellm.completion/acompletion to strip that field right
# before the request goes out. It's safe to remove once CrewAI ships the
# official fix.
# ---------------------------------------------------------------------------
def _strip_cache_breakpoint(messages):
    if not messages:
        return messages
    cleaned = []
    for m in messages:
        if isinstance(m, dict) and "cache_breakpoint" in m:
            m = {k: v for k, v in m.items() if k != "cache_breakpoint"}
        cleaned.append(m)
    return cleaned


if not getattr(litellm, "_cache_breakpoint_patch_applied", False):
    _original_completion = litellm.completion
    _original_acompletion = litellm.acompletion

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        return _original_completion(*args, **kwargs)

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        return await _original_acompletion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
    litellm._cache_breakpoint_patch_applied = True


def build_research_crew(topic: str, groq_api_key: str) -> Crew:
    """Create and return a ready-to-run Crew for the given topic."""

    # --- 1. The LLM (Groq, via CrewAI's built-in LiteLLM routing) ---
    # CrewAI routes "provider/model" strings through LiteLLM automatically.
    # Groq hosts OpenAI's open-weight model under the id "openai/gpt-oss-120b",
    # so the full string CrewAI needs is "groq/openai/gpt-oss-120b".
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.4,
    )

    # --- 2. The tool the agent can use ---
    search_tool = DuckDuckGoSearchTool()

    # --- 3. The single agent ---
    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research the topic '{topic}' thoroughly using web search, "
            "then produce a clear, well-organized, factual report."
        ),
        backstory=(
            "You are an experienced analyst who is skilled at searching the "
            "web, cross-checking facts, and writing reports that are easy "
            "for a general audience to understand. You never make up facts."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,  # single-agent crew: nobody to delegate to
    )

    # --- 4. The single task ---
    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Steps to follow:\n"
            "1. Use the DuckDuckGo Search tool at least twice, with different "
            "search queries, to gather current and reliable information.\n"
            "2. Identify the most important facts, figures, and recent "
            "developments related to the topic.\n"
            "3. Write a well-structured report in Markdown with these "
            "sections, in this order:\n"
            "   - # Title\n"
            "   - ## Introduction\n"
            "   - ## Key Findings (as bullet points)\n"
            "   - ## Analysis\n"
            "   - ## Conclusion\n\n"
            "Do not fabricate facts. If the search results are unclear or "
            "conflicting, say so honestly in the report."
        ),
        expected_output=(
            "A complete Markdown report of at least 400 words on the topic, "
            "with the sections listed above, grounded in the search results."
        ),
        agent=researcher,
    )

    # --- 5. The crew (single agent, single task, runs sequentially) ---
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
