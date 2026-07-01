# GAIA Agent

```bash
pip install -r requirements.txt
export OPEN_ROUTER_API_KEY=your_key
export OPENROUTER_MODEL=z-ai/glm-5.2
export OPENROUTER_SEARCH_MAX_TOKENS=128
export OPENROUTER_ANSWER_MAX_TOKENS=1024
python app.py
```

The agent uses LangGraph for orchestration, OpenRouter for LLM calls, and `ddgs` for web search.

Use the "Benchmark sample" tab to run the agent on the fixed benchmark question.
