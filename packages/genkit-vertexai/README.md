# genkit-vertexai (deprecated)

`genkit-vertexai` is now [`genkit-model-garden`](https://github.com/genkit-ai/genkit-python/tree/main/packages/genkit-model-garden).
This package only re-exports `ModelGarden` from `genkit_model_garden` and warns
on import. It gets no releases for Genkit 1.0.

## Migrate

```bash
uv remove genkit-vertexai
uv add 'genkit-model-garden[anthropic]'   # same [anthropic] and [openai] extras
```

```python
# before
from genkit_vertexai import ModelGarden
# after
from genkit_model_garden import ModelGarden
```

The `ModelGarden` class, its arguments and the `modelgarden/` model names are
unchanged, so saved model refs keep working.

Gemini on Google Cloud was never in this package. It's `VertexAI` in
[`genkit-google-genai`](https://github.com/genkit-ai/genkit-python/tree/main/packages/genkit-google-genai).
