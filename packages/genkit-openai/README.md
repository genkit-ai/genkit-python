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

## Structured output

OpenAI only accepts an object at the root of an output schema, so `OpenAI()`
sends any other schema (a list, enum or scalar, such as `output_schema=int`,
a `str` Enum or `{'type': 'array', 'items': {'type': 'string'}}`) as the
`value` property of an object and unwraps the reply. `response.output` and
the streamed chunks hold the value itself. Requests to Mistral are sent
unwrapped.

## Mistral

`Mistral()` registers Mistral's models and embedders under `mistral/`, as the
JS and Dart SDKs do, so it can sit beside `OpenAI()`:

```python
from genkit import Genkit
from genkit_openai import Mistral, OpenAI

# The Mistral key comes from MISTRAL_API_KEY, the OpenAI one from OPENAI_API_KEY.
ai = Genkit(plugins=[OpenAI(), Mistral()])

res = await ai.generate(
    model=Mistral.model('mistral-small-latest'),  # or 'mistral/mistral-small-latest'
    prompt='Suggest 2 names for a French bakery.',
    config={'temperature': 0.3, 'seed': 42, 'max_output_tokens': 200},
)

embeddings = await ai.embed(embedder='mistral/mistral-embed', content='croissant')
```

`Mistral()` is `OpenAI(provider='mistral')`. `OpenAI(base_url=...)` with one of
Mistral's hosts (`OPENAI_BASE_URL` too) is the same plugin, also named
`mistral`. Its key comes from `api_key=` or `MISTRAL_API_KEY`, never
`OPENAI_API_KEY`. The EU and US endpoints, `https://api.eu.mistral.ai/v1` and
`https://api.us.mistral.ai/v1`, work the same way.

At startup it registers Mistral's own catalog, not OpenAI's:
`mistral-large-latest`, `mistral-large-4`, `mistral-small-latest`,
`ministral-14b-latest`, `ministral-8b-latest`, `ministral-3b-latest`,
`codestral-latest` and `mistral-medium-latest` (deprecated by Mistral), and
the `mistral-embed` (1024 dimensions) and `codestral-embed` (1536) embedders.
The Dev UI's model list comes from Mistral's `/models`, read by each entry's
`capabilities`: ids with `embed` are embedders, entries with `completion_chat`
are chat models, and OCR, moderation, transcription and speech models are left
out. A model outside the catalog takes tool and image support from
`function_calling` and `vision`. Any other id still resolves as a chat model
when named, e.g. `mistral/magistral-medium-latest`.

A gateway in front of Mistral on another host needs `Mistral(base_url=...)`
(or `provider='mistral'`); without it the plugin can't tell the API behind it
is Mistral's and sends OpenAI's field names, which Mistral rejects. Pass the
gateway's own key as `api_key=`, or the plugin reads `MISTRAL_API_KEY`:

```python
Mistral(base_url='https://llm-gateway.example.com/v1', api_key=gateway_key)
```

`Mistral()` with no `base_url` goes to `https://api.mistral.ai/v1`;
`OPENAI_BASE_URL` isn't read then.

Mistral rejects any request field it doesn't declare, so with a Mistral host
(or `provider='mistral'`) the plugin sends Mistral's names for the settings
that differ:

- `max_output_tokens` goes out as `max_tokens`, never `max_completion_tokens`,
  even with `reasoning_effort` set.
- `seed` goes out as `random_seed`.
- `top_k` is refused before the request goes out: Mistral's models reject
  `top_k` sampling. So are `logprobs=True` and any `top_logprobs`: Mistral's
  models answer "Logprobs are not enabled for this model". `logprobs=False`
  is left out of the request.
- `reasoning_effort` also takes `'max'`, which Mistral's API declares and
  OpenAI's doesn't; other targets refuse it before the request goes out.
  Which values a model accepts varies: Mistral Small takes `none` and `high`,
  Mistral Large `low`, `medium` and `high`.
- Streaming sends no `stream_options`; Mistral reports usage on the last chunk
  without it.
- An embedder's `dimensions` goes out as `output_dimension`.
- Media other than images, such as a PDF, is refused before the request goes
  out: every media part goes out as `image_url`. The type comes from the part's
  `content_type`, else a `data:` URL's header; a remote URL with neither is
  sent as an image.
- OpenAI-only settings Mistral doesn't declare are left out of the request,
  with a warning in the log: `verbosity`, `safety_identifier`, `store`,
  `logit_bias`, `prompt_cache_retention`, `web_search_options`, `modalities`
  and `audio`. Mistral would fail the call with a 422 naming them.
  `service_tier`, `prompt_cache_key`, `metadata`, `prediction` and
  `parallel_tool_calls` are Mistral fields too and go out as given
  (`service_tier` takes only `auto` or `standard_only`).
- `extra` gets the same renames: `seed` and `max_completion_tokens` there go
  out as `random_seed` and `max_tokens`, and `stream_options` is dropped.
  Nothing else in `extra` is dropped.

With `reasoning_effort` set, replies carry their reasoning in `thinking`
chunks. Each comes back as its own reasoning part, in the order it came among
the text, with its own signature in `metadata['thoughtSignature']`; a streamed
reply's final message keeps the same order and joins each chunk's fragments
back into one part, ending a part at the fragment that carries a signature. On the next turn each part goes back to Mistral as its own
`thinking` chunk, in order, as Mistral asks, so a signature stays with the text
it signs. Only reasoning that came from Mistral goes back: each such part is
marked `metadata['reasoningSource'] == 'mistral'`, which survives saving the
history as JSON. Reasoning another model left in a mixed history is dropped.

Mistral-only fields like `min_tokens` go in `extra`.

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
