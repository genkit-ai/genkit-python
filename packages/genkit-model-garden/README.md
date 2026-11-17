# Genkit Model Garden Plugin

Call Claude, Llama and other third-party models hosted in Google Cloud
[Model Garden](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-garden/explore-models).
Requests run on your Google Cloud project's credentials and billing, so you
don't need a separate Anthropic or Meta account.

Gemini isn't in this package. For Gemini on Google Cloud, use `VertexAI` from
[`genkit-google-genai`](https://github.com/genkit-ai/genkit-python/tree/main/packages/genkit-google-genai).

This package was published as `genkit-vertexai` through 0.13. Only the package
and import names changed; the `ModelGarden` class and `modelgarden/` model
names are the same.

> **Building with a coding agent? Install the Genkit Python skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-python
> ```
>
> It teaches your agent the current Genkit Python APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Installation

Install the extra for the Model Garden publishers you call:

```bash
uv add 'genkit-model-garden[anthropic]'          # Claude
uv add 'genkit-model-garden[openai]'             # Llama, Mistral, and other OpenAI-compatible models
uv add 'genkit-model-garden[anthropic,openai]'   # both
```

The package without an extra serves no models. Calling a model without its
extra raises an error naming the command to run.

## Usage

```python
from genkit import Genkit
from genkit_model_garden import ModelGarden

ai = Genkit(
    plugins=[ModelGarden(project='my-project', location='us-central1')],
)


async def main() -> None:
    res = await ai.generate(
        model='modelgarden/anthropic/claude-sonnet-4-6',
        prompt='Explain recursion in 10 words.',
    )
    print(res.text)
    # => A function that solves a problem by calling itself repeatedly.


if __name__ == '__main__':
    ai.run_main(main())
```

Authenticates with Google Cloud Application Default Credentials (ADC). The
project comes from `project=`, then `GCLOUD_PROJECT` / `GOOGLE_CLOUD_PROJECT`,
then the ADC project.

Every call runs on those Google Cloud credentials. A per-request key in
`context={'secrets': {'api_key': ...}}` fails with `INVALID_ARGUMENT` before
any request is sent, so a tenant's key never runs on the app's account. Other
entries in `context.secrets` are ignored.
