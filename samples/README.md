# Samples

Each folder runs one feature end to end. Most need `GEMINI_API_KEY`.

```bash
cd samples/<name>
uv sync
uv run src/main.py
```

Web-framework samples (flask-hello, django-hello) list their own server command.

For traces while it runs:

```bash
genkit start -- uv run src/main.py
```

Each folder is named after the feature it shows.

