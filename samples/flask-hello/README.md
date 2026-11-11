# Flask

Hang a flow on the Flask app you already have.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

```bash
curl -N -X POST http://localhost:8080/say_hi \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"data":"Mittens"}'
```
