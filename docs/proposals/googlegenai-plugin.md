# Proposal: One `GoogleGenAI` plugin with an `enterprise=` flag

**Author:** Jeff Huang · **Status:** Proposed · **Applies to:** `genkit-google-genai` (Python), before 1.0

## Summary

`genkit-google-genai` has two plugins for Gemini: `GoogleAI` (an AI Studio API key) and `VertexAI` (a Google Cloud project). They call the same models. The only difference is which Google API serves them.

This proposal replaces both with one plugin, named after the package, before Python 1.0:

```python
GoogleGenAI()                                                                # AI Studio key
GoogleGenAI(enterprise=True, project='bistro-prod', location='us-central1')  # Google Cloud
```

`GoogleAI` and `VertexAI` are removed, not deprecated. Python 1.0 ships with `GoogleGenAI` only, so there's no deprecation path to carry and no alias machinery to build.

## Why

**Going to production shouldn't mean rewriting model names.** Today a team prototypes with `googleai/gemini-2.5-flash` on an AI Studio key. Moving to their Cloud project means changing every ref to `vertexai/gemini-2.5-flash`: in code, in `.prompt` files and in eval datasets. With one plugin, they flip one flag at setup and the refs stay the same.

**The package name outlives the product name.** Vertex AI became Gemini Enterprise Agent Platform in April, then "Gemini platform" in October. `google-genai` is the package name on PyPI and npm, and the name of the SDK underneath. A plugin named after it doesn't move when marketing does.

**The SDK underneath already works this way.** `google-genai` uses one client with a backend switch: `Client(enterprise=True)`, with `vertexai=` documented as legacy.

**Before 1.0 is the cheap time to do it.** After 1.0, removing `GoogleAI` and `VertexAI` needs a deprecation cycle, a `vertexai/` alias in core, and a 2.0. Before 1.0 it's a rename.

## Design

```python
# 1. Prototype on an AI Studio key
ai = Genkit(plugins=[GoogleGenAI()])

# 2. Production on a Cloud project: same refs, same .prompt files
ai = Genkit(plugins=[GoogleGenAI(enterprise=True, project='bistro-prod', location='us-central1')])

# 3. One ref helper; the plugin decides the backend
response = await ai.generate(model=GoogleGenAI.gemini('gemini-2.5-flash'), prompt='Suggest a dish.')
# => runs on whichever backend the plugin was configured for

# 4. A model the configured backend doesn't serve fails at lookup
await ai.generate(model=GoogleGenAI.virtual_try_on('virtual-try-on-001'), ...)  # on the default backend
# => GenkitError: NOT_FOUND, 'googlegenai/virtual-try-on-001' needs GoogleGenAI(enterprise=True)
```

- **`enterprise=` is keyword-only and defaults to `False`.** `GoogleGenAI()` behaves like `GoogleAI()` today, and matches `genai.Client()`, which also defaults to the Developer API. The default can't bill a Cloud project by accident: Cloud needs an explicit `enterprise=True`, and without a Gemini API key the default fails at setup with an error naming the key.
- **The backend is set on the plugin only.** It isn't in `GeminiConfig` or any model config. Config is merged from the ref, the `.prompt` file and the call, and middleware can rewrite it, so a config field would let any of those move an app to a different billing account. It also couldn't work: `resolve()` and the Dev UI model list run before any request exists, and need to know the backend to know which models exist.
- **One namespace: `googlegenai/`.** Typed refs are classmethods (`GoogleGenAI.gemini('x')`). They run without a plugin instance, so they can't know the backend and need one fixed prefix. The ref names the model; the plugin picks the backend.
- **Traces record the backend.** The ref no longer shows which account paid, so the backend goes in span metadata.
- **One app, one backend.** Two `GoogleGenAI` plugins collide on `googlegenai/`, and the registry raises at startup, with a message that says so. Running `GoogleAI()` and `VertexAI()` side by side, as you can today, is no longer supported. We accept that.

## How it's built

The request code is already shared. `GoogleAI` and `VertexAI` build the same Gemini, Veo and embedder model classes, and those read the backend off the `genai.Client` they're handed (`_is_vertex()` returns `client.vertexai`). That one flag already picks the endpoint and auth, and `_check_backend_fields` already rejects the other backend's config fields before anything is sent.

What's split today is the two plugin classes (~500 lines each): setup and auth, which models `resolve()` and `list_actions()` offer, and the typed ref classmethods. `GoogleGenAI` folds those into one class, keyed on `enterprise`:

- **Setup and auth** pick the `GoogleAI` or `VertexAI` path from the flag.
- **`resolve()` and `list_actions()`** serve the catalog for the configured backend. A backend-only model on the wrong backend gets a `NOT_FOUND` that names the backend, instead of today's generic not-found.
- **Typed refs** all live on `GoogleGenAI`: `.gemini()`, `.gemini_tts()`, `.gemini_image()`, `.gemma()`, `.veo()`, `.embedder()`, `.deep_research()`, `.antigravity()`, `.lyria()`, `.virtual_try_on()`. Known-name lists become the union of both backends.
- **Internals:** we pass `enterprise=` to `genai.Client`, and internal names move from `vertex` to `enterprise`.

## Constructor settings

`enterprise=True` uses exactly what `VertexAI` does today, and `enterprise=False` uses exactly what `GoogleAI` does today.

```python
# before
GoogleAI(api_key=key)
VertexAI(project='bistro-prod', location='us-central1')
VertexAI(api_key=key)                                      # Cloud express mode

# after
GoogleGenAI(api_key=key)
GoogleGenAI(enterprise=True, project='bistro-prod', location='us-central1')
GoogleGenAI(enterprise=True, api_key=key)                  # Cloud express mode
```

- **`enterprise=True`:**
  - `project`: the argument, then `GCLOUD_PROJECT` / `GOOGLE_CLOUD_PROJECT`, then `credentials.project_id`, then the ADC project.
  - `location`: the argument, then `GOOGLE_CLOUD_LOCATION` / `GCLOUD_LOCATION`, then `us-central1`. Multi-regions keep their dedicated endpoints.
  - Express mode only with an explicit `api_key=`. Environment keys stay ignored, as they are today, so an AI Studio `GEMINI_API_KEY` never quietly becomes a Cloud key.
- **`enterprise=False` (default):** the key comes from `api_key`, then `GEMINI_API_KEY`, then `GOOGLE_API_KEY`. Passing `project` or `location` raises a `ValueError` at setup.
- **Only `enterprise=` picks the backend.** The SDK reads `GOOGLE_GENAI_USE_ENTERPRISE` and `GOOGLE_GENAI_USE_VERTEXAI` only when no flag is passed. The plugin always passes one, so both variables stay ignored, as they are today.

## What differs between backends

The refs stay the same, but what each backend serves doesn't:

- **Developer API only:** Lyria, Deep Research, Antigravity, `file_search_*`.
- **Cloud only:** Virtual Try-On, tuned Gemini endpoints, multimodal embedders, image `output_mime_type`/`output_compression_quality`, `person_generation`, `prominent_people`, `google_search_exclude_domains`/`blocking_confidence`, `safety_settings[].method`, per-request `location`, embedder `mime_type`/`auto_truncate`.
- **Veo:** the IDs don't overlap. The Developer API uses `veo-3.1-generate-preview`, and Cloud uses `veo-3.1-generate-001`.
- **Gemini:** most names are shared. The `-latest` aliases (`gemini-flash-latest`, …) exist only on the Developer API.

A backend-only model fails at lookup, and a backend-only config field fails before the request is sent. Both errors name the backend and the fix.

## Interactions API

Gemini can be called through `generateContent` (each request sends the full conversation) or the newer Interactions API (`/interactions`, with server-side `store` and background runs). Users don't pick between them; the plugin chooses per model.

Today only Deep Research, Antigravity and Lyria use Interactions, through the plugin's own client, and only on the Developer API. That doesn't block this proposal: with `enterprise=True`, those models simply aren't offered, the same as on `VertexAI` today.

It matters when Gemini itself moves onto Interactions, after [#78](https://github.com/genkit-ai/genkit-python/pull/78) and [#105](https://github.com/genkit-ai/genkit-python/pull/105). Cloud serves Interactions at `v1beta1/projects/{p}/locations/{l}/interactions` with OAuth (express mode: `locations/global/interactions`), and rejects `top_k` and image models there. So the ask for that work: the Interactions client takes the backend and credentials as values passed in, not inferred from the plugin class. Then `GoogleGenAI` wires `enterprise=` into both code paths, and the two code paths can't end up on different backends.

## Migration

Python is pre-1.0, so this is a breaking rename with a release note:

```python
# before
from genkit_google_genai import GoogleAI, VertexAI
ai = Genkit(plugins=[GoogleAI()])
await ai.generate(model=GoogleAI.gemini('gemini-2.5-flash'), prompt='...')
await ai.generate(model='googleai/gemini-2.5-flash', prompt='...')

# after
from genkit_google_genai import GoogleGenAI
ai = Genkit(plugins=[GoogleGenAI()])
await ai.generate(model=GoogleGenAI.gemini('gemini-2.5-flash'), prompt='...')
await ai.generate(model='googlegenai/gemini-2.5-flash', prompt='...')
```

In this repo, that's about 100 references to `GoogleAI`/`VertexAI` outside the plugin, 83 `googleai/` or `vertexai/` ref strings, and 3 `.prompt` files, mostly samples. It's mechanical.

## Other SDKs

This proposal covers Python. JS agreed on the same shape for JS 2.0: one `googleGenAI` plugin with an `enterprise` flag. JS discussed making the flag required; Python defaults it to `False`, and the two SDKs should land on the same answer. In JS, the parent can hold the existing `googleAI` or `vertexAI` plugin internally and delegate to it, so the change there is a layer on top rather than a rewrite.

One thing to keep in step: `.prompt` files are shared across SDKs, so the namespace string has to match. If Python registers `googlegenai/`, JS 2.0 and Go should too. A JS parent that delegates to its inner plugin should still register its models under `googlegenai/`, not `googleai/` or `vertexai/`.

## Plan

1. **Now:** sign-off on this proposal from Python and JS owners.
2. **Before Python 1.0, one PR:**
   - `GoogleGenAI` with `enterprise` (default `False`), `project` and `location`;
   - catalog and `resolve()` keyed on `enterprise`, with backend-named `NOT_FOUND`;
   - the backend in span metadata;
   - remove `GoogleAI` and `VertexAI` exports;
   - update samples, READMEs and `.prompt` files.
3. **Independent:** [genkit-python#137](https://github.com/genkit-ai/genkit-python/pull/137), the Model Garden rename. It's also breaking, so it lands before 1.0 too.
4. **After #78 and #105:** Gemini on Interactions, with the Cloud transport taking the backend as a value.

## Open question

1. **`.prompt` files that still say `googleai/` or `vertexai/`.** Until JS and Go ship their major versions, a `.prompt` file written for them won't resolve in Python 1.0.
   - **No aliases:** these refs return `NOT_FOUND`. Clean, and the release note covers it, but shared `.prompt` files break until the other SDKs catch up.
   - **Aliases:** Python resolves `googleai/` when `enterprise=False` and `vertexai/` when `enterprise=True`, to the same `googlegenai/` action. Shared files keep working, at the cost of a small core hook (plugins claiming extra namespace names) before 1.0.
