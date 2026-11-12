# Context caching

Cache a handbook once. Follow-up questions skip those tokens.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

Each call prints `cached_content_tokens`: the first call creates the cache,
and every question after it sends only its own tokens.
