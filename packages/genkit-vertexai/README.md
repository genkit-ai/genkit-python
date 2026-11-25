# Genkit Vertex AI Plugin

Integrate Genkit with Google Cloud Vertex AI Model Garden.

> **Claude through `ModelGarden` is deprecated.** Use the Anthropic plugin on
> Google Cloud instead; see [Moving Claude to the Anthropic plugin](#moving-claude-to-the-anthropic-plugin).
> `modelgarden/anthropic/...` models keep working in this release, warn the
> first time each one is used, and will be removed in the next release.

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
uv add 'genkit-vertexai[openai]'             # Llama, Mistral, and other OpenAI-compatible models
uv add 'genkit-vertexai[anthropic]'          # Claude (deprecated)
```

The package without an extra serves no models. Calling a model without its
extra raises an error naming the command to run.

## Usage

```python
from genkit import Genkit
from genkit_vertexai import ModelGarden

ai = Genkit(
    plugins=[ModelGarden(project='my-project', location='us-central1')],
)


async def main() -> None:
    res = await ai.generate(
        model='modelgarden/meta/llama-3.1-405b-instruct-maas',
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

## Moving Claude to the Anthropic plugin

Claude on Google Cloud now runs through `genkit-anthropic`, billed to the same
project with the same credentials. You get typed refs and `AnthropicConfig`,
and `.prompt` files that also work on the Anthropic API.

```bash
uv add 'genkit-anthropic[google-cloud]'
```

```python
from genkit import Genkit
from genkit_anthropic import Anthropic, GoogleCloud

# Before
ai = Genkit(plugins=[ModelGarden(project='bistro-prod', location='global')])
await ai.generate(model='modelgarden/anthropic/claude-sonnet-5-5', prompt='Suggest a dish.')

# After
ai = Genkit(plugins=[Anthropic(backend=GoogleCloud(project='bistro-prod', location='global'))])
await ai.generate(model=Anthropic.claude('claude-sonnet-5-5'), prompt='Suggest a dish.')
```

- Model names drop the `modelgarden/anthropic/` prefix and any `@date`
  suffix: `modelgarden/anthropic/claude-sonnet-4-5@20250929` becomes
  `Anthropic.claude('claude-sonnet-4-5')`.
- `project` and `location` mean the same thing in `GoogleCloud(...)` and
  default the same way. `model_locations` has no equivalent; use one
  `Anthropic(backend=GoogleCloud(location=...))` per region.
- Config is unchanged: Model Garden Claude already took `AnthropicConfig`.
- Keep `ModelGarden` for Llama and the other OpenAI-compatible models.

See the [genkit-anthropic README](../genkit-anthropic/README.md#claude-on-google-cloud)
for first-time Google Cloud setup.
