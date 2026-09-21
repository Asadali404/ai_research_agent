# 🔎 AI Research Agent

A single-agent [CrewAI](https://docs.crewai.com) app that researches any topic you give it
using free DuckDuckGo web search, then writes a structured Markdown report — powered by
[Groq](https://groq.com)'s `openai/gpt-oss-120b` model, with a [Streamlit](https://streamlit.io) frontend.

## How it works

- **1 agent** — a "Senior Research Analyst"
- **1 tool** — a custom DuckDuckGo search tool (no API key needed, `ddgs` package)
- **1 task** — search the topic, then write the report
- **1 LLM** — Groq's `openai/gpt-oss-120b`, called through CrewAI's built-in LiteLLM routing
- **API key** — read only from Streamlit's built-in Secrets manager (`st.secrets`)

```
ai-research-agent/
├── app.py                          # Streamlit UI — entry point
├── research_crew.py                # Defines the Agent, Task, and Crew
├── tools/
│   └── search_tool.py              # Custom DuckDuckGo search tool
├── requirements.txt                # Pinned, compatible dependency versions
├── .gitignore
└── README.md
```

---

## Step 1 — Get a free Groq API key

1. Go to <https://console.groq.com/keys>
2. Sign up (it's free)
3. Click **"Create API Key"** and copy it — you'll only see it once, so save it somewhere safe for now.

---

## Step 2 — Upload this project to GitHub

1. Go to <https://github.com/new> and create a new **empty** repository
   (skip adding a README there — this project already has one).
2. Open your new repo and click **"uploading an existing file"** (or the
   **Add file → Upload files** button).
3. Drag in every file and folder from this project — keep `tools/search_tool.py`
   inside a `tools` folder, exactly as shown in the structure above.
4. Scroll down and click **Commit changes**.

That's it — no `git` commands needed. Your API key is never part of this
upload, since it isn't stored in any file here.

---

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **"New app"**.
3. Pick your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Before clicking Deploy, open **"Advanced settings" → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_real_groq_api_key_here"
   ```
   (paste the key you copied in Step 1)
5. Click **Deploy**. The first build takes a few minutes while it installs
   dependencies.
6. Once it's live, open the app — type a topic and click **Run Research**.
   The first run usually takes 30–90 seconds.

If you ever need to update the key later: go to your app on
[share.streamlit.io](https://share.streamlit.io) → **Settings → Secrets**,
edit it, and the app will restart automatically.

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| App shows "GROQ_API_KEY is not set..." | You skipped or mistyped the Secrets step. Go to Settings → Secrets on Streamlit Cloud and add it exactly as `GROQ_API_KEY = "..."`. |
| Build fails on `ModuleNotFoundError: No module named 'ddgs'` | Make sure `requirements.txt` was actually uploaded to the repo root — check the file list on GitHub. |
| Empty/failed search results | DuckDuckGo occasionally rate-limits; wait a few seconds and retry, or try a more specific topic. |
| `AuthenticationError` / 401 from Groq | Your API key is wrong or has been revoked — generate a new one at <https://console.groq.com/keys> and update it in Streamlit Cloud Secrets. |
| App builds but errors immediately when clicking Run | Open **Manage app → Logs** on Streamlit Cloud to see the full Python traceback. |

## Next steps to extend this
- Add a second agent (e.g. an "editor" that polishes the report) and a second `Task`.
- Swap `openai/gpt-oss-120b` for another Groq-hosted model (see the [Groq model list](https://console.groq.com/docs/models)) by editing the `model=` string in `research_crew.py`.
- Add an `output_file` to the `Task` so every report also saves to disk automatically.
