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

On Chat Completions, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.5`,
`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-6-sol` and `gpt-6-luna`
accept function tools only with `reasoning_effort='none'`. The `gpt-5.4`
models and `gpt-5.5` default to `none`. The others do not, so set
`reasoning_effort='none'` in `config` when you pass them tools. `gpt-6-astra`, `gpt-6.1-sol` and `gpt-6.1-terra` do not advertise
tool support, because they take function tools only through the Responses API.
`gpt-5-search-api` does not take function tools either.

The Dev UI lists the models your key can see on `/v1/models`, but only the ones
this plugin can call: realtime, live, audio, moderation, Responses-only and
legacy completion models are left out. A dated snapshot such as
`gpt-5-2025-08-07` takes the capabilities and stage of its alias, and a
fine-tune `ft:<base>:...` takes those of its base model.

`KnownGpt` lists only the chat models that OpenAI serves today. Ids that OpenAI
has shut down (for example `gpt-4-turbo`, `gpt-3.5-turbo`, `o1` and `o4-mini`)
are no longer in it, so `OpenAI.gpt('o4-mini')` fails a type check. These ids
still resolve by name with their old capabilities, for hosts on an
OpenAI-compatible `base_url` that still serve them.

## Config

`config` takes the settings `OpenAIConfig` declares. Any other key raises
before the request is sent, so a typo fails by name:

```python
res = await ai.generate(
    model=OpenAI.gpt('gpt-5.5'),
    prompt='Write a haiku about the sea.',
    config={
        'reasoning_effort': 'low',
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
    model=OpenAI.gpt('gpt-5.5'),
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
