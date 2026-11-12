# Genkit OpenAI Plugin

OpenAI-compatible model provider for Genkit (OpenAI, Azure OpenAI, and other
compatible endpoints).

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

```bash
uv add genkit-openai
```

## Usage

```python
from genkit import Genkit
from genkit_openai import OpenAI

ai = Genkit(plugins=[OpenAI()])


async def main() -> None:
    res = await ai.generate(
        model=OpenAI.gpt('gpt-5.2'),
        prompt='Suggest 2 catchy names for an AI newsletter.',
    )
    print(res.text)
    # => 1. The Neural Brief
    #    2. Prompt & Circumstance


if __name__ == '__main__':
    ai.run_main(main())
```

Set `OPENAI_API_KEY` in the environment, or pass `api_key=` to `OpenAI()`.
`base_url=` points the plugin at another OpenAI-compatible server. Pass the URL
up to and including `/v1`; the SDK appends `/chat/completions`. Any other
`AsyncOpenAI` setting goes in `client_options`:

```python
OpenAI(
    api_key=key,
    base_url='https://openrouter.ai/api/v1',
    client_options={'organization': 'org-123', 'timeout': 30, 'max_retries': 5},
)
```

## Mistral

Point `base_url` at Mistral's API and pass a Mistral key. Models keep the
`openai/` prefix:

```python
import os

from genkit import Genkit
from genkit_openai import OpenAI

ai = Genkit(
    plugins=[OpenAI(api_key=os.environ['MISTRAL_API_KEY'], base_url='https://api.mistral.ai/v1')]
)

res = await ai.generate(
    model='openai/mistral-small-latest',
    prompt='Suggest 2 names for a French bakery.',
    config={'temperature': 0.3, 'seed': 42, 'max_output_tokens': 200},
)

embeddings = await ai.embed(embedder='openai/mistral-embed', content='croissant')
```

Pass `api_key=` explicitly. Without it the plugin falls back to
`OPENAI_API_KEY` and sends that key to Mistral. The EU and US endpoints,
`https://api.eu.mistral.ai/v1` and `https://api.us.mistral.ai/v1`, work the
same way.

Mistral rejects any request field it doesn't declare, so with a Mistral host
the plugin sends Mistral's names for the settings that differ:

- `max_output_tokens` goes out as `max_tokens`, never `max_completion_tokens`,
  even with `reasoning_effort` set.
- `seed` goes out as `random_seed`.
- `top_k` goes out as is. Mistral takes it; OpenAI never gets it.
- Streaming sends no `stream_options`; Mistral reports usage on the last chunk.
- An embedder's `dimensions` goes out as `output_dimension`.

With `reasoning_effort` set, replies carry their reasoning in `thinking`
chunks. Those come back as reasoning parts, ahead of the text, streamed or not,
and go back to Mistral as a `thinking` chunk on the next turn, as Mistral asks.

Other OpenAI-only settings, such as `verbosity` or `store`, are sent as given
and Mistral fails the call naming them. Mistral-only fields like `min_tokens`
go in `extra`. A gateway in front of Mistral on any other host gets the OpenAI
field names.

The plugin doesn't know Mistral's embedders, so their info reports the default
1536 dimensions; `mistral-embed` returns 1024.

## Config

`config` takes the settings `OpenAIConfig` declares. Any other key raises
before the request is sent, so a typo fails by name:

```python
res = await ai.generate(
    model=OpenAI.gpt('gpt-4o'),
    prompt='Write a haiku about the sea.',
    config={
        'temperature': 0.2,
        'max_output_tokens': 200,  # caps reply length
        'extra': {'user': 'diner-42'},  # a field OpenAIConfig doesn't declare
    },
)
```

- `max_output_tokens` (`maxOutputTokens` in the Dev UI) caps reply length.
  It goes out as `max_tokens`, or as `max_completion_tokens` on reasoning
  models and whenever `reasoning_effort` is set. `stop_sequences` goes out as
  `stop`. OpenAI's own names (`max_tokens`, `max_completion_tokens`, `stop`)
  are not declared on `OpenAIConfig`, so passing them at the top level raises
  `INVALID_ARGUMENT`.
- `extra` goes out as top-level request body fields. A key in `extra`
  replaces a setting of the same name (including `max_tokens` or
  `max_completion_tokens` for OpenAI-compatible servers). An API key in
  `config` or `extra` raises `INVALID_ARGUMENT`; it belongs in
  `context.secrets`.

## Per-request API key

To bill a call to someone else's OpenAI account, pass their key in
`context.secrets`. It's used for that call only, on every model and embedder
the plugin serves:

```python
res = await ai.generate(
    model=OpenAI.gpt('gpt-4o'),
    prompt='hi',
    context={'secrets': {'api_key': tenant_key}},
)
```

- `apiKey` works as well as `api_key`. Other entries in `context.secrets`
  are left alone, so a call whose secrets hold no key runs on the plugin's
  key. A key that is blank or not a string raises `INVALID_ARGUMENT`.
- The tenant call sends only the tenant's key. The plugin's `organization`,
  `project`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` are left off; `base_url`,
  timeouts and other `default_headers` carry over. A plugin whose
  `default_headers` pin `Authorization` refuses a tenant key with
  `FAILED_PRECONDITION`.
- The plugin doesn't need a key of its own: `OpenAI()` with no
  `OPENAI_API_KEY` serves every `generate()` or `embed()` call that brings a
  key in `context.secrets`, and a call without one fails with
  `FAILED_PRECONDITION`.
- Embedders read the same key:
  `ai.embed(..., context={'secrets': {'api_key': tenant_key}})`. A call
  without one runs on the plugin's key.
