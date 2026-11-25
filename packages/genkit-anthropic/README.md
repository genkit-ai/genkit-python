# Genkit Anthropic Plugin

Anthropic Claude model provider for Genkit.

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
uv add genkit-anthropic
```

## Usage

```python
from genkit import Genkit
from genkit_anthropic import Anthropic

ai = Genkit(plugins=[Anthropic()])


async def main() -> None:
    res = await ai.generate(
        model=Anthropic.claude('claude-sonnet-4-6'),
        prompt='Explain recursion in 10 words.',
    )
    print(res.text)
    # => A function that solves a problem by calling itself repeatedly.


if __name__ == '__main__':
    ai.run_main(main())
```

Set `ANTHROPIC_API_KEY` in the environment, or pass `api_key=` to `Anthropic()`.
Create a key in the [Anthropic Console](https://console.anthropic.com/settings/keys).
To run Claude through Google Cloud instead, see
[Claude on Google Cloud](#claude-on-google-cloud).
`base_url=` points the plugin at another server. Pass the origin with no `/v1`
suffix; the SDK appends `/v1/messages`. Any other `AsyncAnthropic` setting goes
in `client_options`:

```python
Anthropic(
    api_key=key,
    base_url='https://claude-proxy.example.com',
    client_options={'timeout': 30, 'max_retries': 5, 'default_headers': {'X-Team': 'search'}},
)
```

## Claude on Google Cloud

You can also run the same Claude models through Google Cloud. Usage is billed
to your Google Cloud project instead of an Anthropic account, and you
authenticate with your Google login instead of an API key. Your code doesn't
change apart from one constructor argument: refs, `AnthropicConfig` and
`.prompt` files work the same on both.

```python
# Anthropic API (ANTHROPIC_API_KEY)
Anthropic()

# Google Cloud (your gcloud login and project)
Anthropic(backend=GoogleCloud())
```

Most users of this plugin call the Anthropic API, so `Anthropic()`'s own
arguments (`api_key`, `base_url`) are about that. Everything Google Cloud needs
goes in the one `GoogleCloud(...)` object instead of adding more arguments to
`Anthropic()`.

### First-time setup

This takes about five minutes the first time. If you already use Vertex AI
(now Gemini Enterprise Agent Platform) and have Claude enabled in a project,
skip to step 5.

1. **Install the gcloud CLI** ([install guide](https://cloud.google.com/sdk/docs/install)),
   then log in:

   ```bash
   gcloud auth login
   ```

2. **Pick a project with billing turned on.** To make a new one, use
   [Create a project](https://console.cloud.google.com/projectcreate), then
   link a billing account on the [Billing page](https://console.cloud.google.com/billing).
   Then make it your default project:

   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Turn on the API** that serves Claude on Google Cloud:

   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

4. **Enable Claude for the project.** Open the model's card in Model Garden,
   e.g. [Claude Sonnet 4.6](https://console.cloud.google.com/agent-platform/publishers/anthropic/model-garden/claude-sonnet-4-6),
   click **Enable**, and accept Anthropic's terms. You do this once per model,
   per project. If you didn't create the project yourself, ask its owner for
   the *Consumer Procurement Entitlement Manager* role (to enable models) and
   the *Agent Platform User* role (to call them).

5. **Give your code access to your login:**

   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project YOUR_PROJECT_ID
   ```

   The second command is optional. It stops google-auth from warning that
   your login has no quota project.

6. **Install the extra and call Claude:**

   ```bash
   uv add 'genkit-anthropic[google-cloud]'
   ```

   ```python
   from genkit import Genkit
   from genkit_anthropic import Anthropic, GoogleCloud

   # 1. Project comes from `gcloud config set project`; location defaults to 'global'
   ai = Genkit(plugins=[Anthropic(backend=GoogleCloud())])


   async def main() -> None:
       # 2. Same typed ref and config as the Anthropic API
       res = await ai.generate(
           model=Anthropic.claude('claude-sonnet-4-6'),
           prompt='Suggest a dish for a guest with a shellfish allergy.',
       )
       print(res.text)
       # => Herb-Roasted Chicken with Lemon Rice


   if __name__ == '__main__':
       ai.run_main(main())
   ```

### Deploying to Google Cloud

On Cloud Run, GKE or Compute Engine, skip steps 1 and 5. The code runs as the
service's service account, and `GoogleCloud()` picks up the project from the
environment. The service account needs the *Agent Platform User* role:

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member=serviceAccount:YOUR_SERVICE_ACCOUNT_EMAIL \
    --role=roles/aiplatform.user
```

### If a call fails

The error comes back on `response.error`, and the Dev UI shows it on the run.

- **`FAILED_PRECONDITION` ... found no Google Cloud credentials**, or
  **... could not refresh Google Cloud credentials**: run step 5 again. The
  next call picks up the new login; you don't need to restart the app.
- **`FAILED_PRECONDITION` ... needs a Google Cloud project**: run
  `gcloud config set project YOUR_PROJECT_ID`, set `GOOGLE_CLOUD_PROJECT`, or
  pass `GoogleCloud(project='YOUR_PROJECT_ID')`.
- **`PERMISSION_DENIED`, ... API has not been used in project ... or it is
  disabled**: run step 3. It can take a few minutes to take effect.
- **`PERMISSION_DENIED` naming `aiplatform.endpoints.predict`**: your login or
  service account needs the *Agent Platform User* role (step 4, or
  [Deploying to Google Cloud](#deploying-to-google-cloud)).
- **`NOT_FOUND` for the model**: enable that model for this project (step 4),
  and check the model id.
- **`RESOURCE_EXHAUSTED`**: the project is out of quota for that model. Check
  the model's quota on the [Quotas page](https://console.cloud.google.com/iam-admin/quotas),
  and request more there if you need it.

### Settings

- `project` is taken from, in order: `GoogleCloud(project=...)`,
  `GCLOUD_PROJECT`, `GOOGLE_CLOUD_PROJECT`, then your credentials (your
  `gcloud config` project locally, the service's project on Google Cloud).
- `location` is taken from `GoogleCloud(location=...)`, then
  `GOOGLE_CLOUD_LOCATION`, then `global`. Every project that enables Claude
  gets quota on the `global` endpoint; regional endpoints need regional quota.
- Only `backend=` switches to Google Cloud. A missing `ANTHROPIC_API_KEY`
  never falls back to it, and `api_key=` with `backend=GoogleCloud()` raises
  `ValueError`.
- `client_options` go to `AsyncAnthropicVertex`. Pass `credentials=` or
  `access_token=` there to use your own credential instead of your gcloud
  login; then set the project with `project=` or an environment variable.
- Google Cloud has no Models API, so the Dev UI lists the plugin's built-in
  Claude catalog. Any other Claude id still resolves when you call it.
- Image URLs aren't supported on Google Cloud (base64 images work), and an
  organization policy can block structured output or web search for partner
  models.

## Typed config

`AnthropicConfig` is flat. Choices are exported Literals (`ThinkingMode`,
`ThinkingDisplay`, `Effort`), so pyright, pyrefly and ty check every key and
value. The plugin builds Anthropic's nested request objects from the flat
fields:

- `thinking`, `thinking_budget`, `thinking_display` → `thinking`
- `effort`, `task_budget` → `output_config`
- `user_id` → `metadata`

```python
from genkit_anthropic import AnthropicConfig

# 1. Build the config from flat fields
config = AnthropicConfig(
    max_output_tokens=4096,
    thinking='adaptive',
    thinking_display='summarized',
    effort='high',
    task_budget=20000,
    user_id='diner-42',
)

# 2. Generate with it
await ai.generate(
    model=Anthropic.claude('claude-opus-4-6'),
    prompt='Plan a tasting menu.',
    config=config,
)
# Sent as:
#   thinking={'display': 'summarized', 'type': 'adaptive'}
#   output_config={'effort': 'high', 'task_budget': {'type': 'tokens', 'total': 20000}}
#   metadata={'user_id': 'diner-42'}
```

- A `thinking_budget` alone means `thinking='enabled'`. `'enabled'` needs a
  budget from some config layer; `'adaptive'` doesn't send one, and
  `'disabled'` sends neither the budget nor the display.
- `task_budget` is a beta field, so it sends the call to the beta API.
- To send a shape the flat fields don't cover, put Anthropic's object in
  `extra`, e.g. `extra={'thinking': {...}}`. It replaces the built object.

### Defaults and overrides

Config merges key by key: the `Anthropic.claude(config=...)` default, then the
prompt's config, then the call's. Rules that span fields are checked once, on
the merged config. `thinking` is the switch, so a call can turn thinking off
or back on without restating the budget:

```python
# 1. Model default with extended thinking on
claude = Anthropic.claude('claude-sonnet-4-6', config=AnthropicConfig(thinking='enabled', thinking_budget=4096))

# 2. Turn it off for one call; the inherited budget isn't sent
await ai.generate(model=claude, prompt='Is shellfish safe for this guest?', config={'thinking': 'disabled'})
# Sent as: thinking={'type': 'disabled'}

# 3. Raise the budget for another; the mode is inherited
await ai.generate(model=claude, prompt='Plan a nut-free tasting menu.', config={'thinking_budget': 8192})
# Sent as: thinking={'type': 'enabled', 'budget_tokens': 8192}
```

A budget over an inherited `thinking='disabled'` stays disabled. Pass
`thinking='enabled'` to turn it on, or `thinking=None` to clear the inherited
mode. `None` clears any inherited value the same way.

## Tool choice

Tool choice is Genkit's `tool_choice` option, not a config field. The plugin
sends it as Anthropic's `tool_choice` object: `'auto'` → `{'type': 'auto'}`,
`'required'` → `{'type': 'any'}`, `'none'` → `{'type': 'none'}`. Anthropic
carries `disable_parallel_tool_use` inside that object, so the config takes it
as a flat flag:

```python
# 1. Require one tool call, and only one
res = await ai.generate(
    model='anthropic/claude-sonnet-4-6',
    prompt='Is the pho in stock?',
    tools=['lookup_menu'],
    tool_choice='required',
    config=AnthropicConfig(disable_parallel_tool_use=True),
    return_tool_requests=True,
)
# Sent as: tool_choice={'type': 'any', 'disable_parallel_tool_use': True}
```

To pin one named tool, send Anthropic's object as
`extra={'tool_choice': {'type': 'tool', 'name': 'lookup_menu'}}`; it replaces
the translated value.

With `thinking='enabled'` or `'adaptive'`, Anthropic accepts only `'auto'` and
`'none'`, so `tool_choice='required'` raises `INVALID_ARGUMENT` before the
request goes out. On a request with no tools, `tool_choice` (option or
`extra`) is dropped, since the API rejects it there.

## Per-request API key

To bill a call to someone else's Anthropic account, pass their key in
`context.secrets`. It's used for that call only:

```python
res = await ai.generate(
    model='anthropic/claude-sonnet-4-6',
    prompt='hi',
    context={'secrets': {'api_key': tenant_key}},
)
```

- `apiKey` works as well as `api_key`. Other entries in `context.secrets`
  are left alone, so a call whose secrets hold no key runs on the plugin's
  key. A key that is blank or not a string raises `INVALID_ARGUMENT`.
- A key in `config` or `config.extra` raises `INVALID_ARGUMENT`; config is
  recorded in traces.
- The plugin doesn't need a key of its own: `Anthropic()` with no
  `ANTHROPIC_API_KEY` serves every call that brings a key in
  `context.secrets`, and a call without one fails with `FAILED_PRECONDITION`.
- A client whose credential can't be swapped refuses a tenant key with
  `FAILED_PRECONDITION` rather than billing its own account: an
  `Anthropic(client_options={'auth_token': ...})` client, one whose `default_headers` pin
  `x-api-key` or `Authorization`, and `Anthropic(backend=GoogleCloud())`, which
  authenticates with Google Cloud credentials.

## Disclaimer

Use of Anthropic's API is subject to
[Anthropic's Terms of Service](https://www.anthropic.com/terms) and
[Privacy Policy](https://www.anthropic.com/privacy). You are responsible for
complying with all applicable terms when using this plugin.

- **API Key Security** — Never commit your Anthropic API key to version control.
  Use environment variables or a secrets manager.
- **Usage Limits** — Be aware of your Anthropic plan's rate limits and token
  quotas. See [Anthropic Pricing](https://www.anthropic.com/pricing).
- **Data Handling** — Review Anthropic's data processing practices before
  sending sensitive or personally identifiable information.

## License

Apache-2.0
