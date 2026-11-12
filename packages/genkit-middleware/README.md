# Genkit Middleware Plugin

A collection of middleware implementations for Genkit Python.

> **Building with a coding agent? Install the Genkit Python skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-python
> ```
>
> It teaches your agent the current Genkit Python APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Overview

This plugin provides concrete middleware implementations for common use cases:

- **Retry**: Retries model API calls on transient errors with exponential backoff
- **Fallback**: Falls back to alternative models when the primary model fails
- **ToolApproval**: Requires explicit approval before executing tool calls
- **Skills**: Exposes a library of skills as system prompts and tools
- **Filesystem**: Provides sandboxed filesystem operations
- **Artifacts** (experimental, `genkit_middleware.exp`): Session artifact listing plus read/write artifact tools

## Quick start

Import the middleware classes you need and pass instances directly into `use=[]`:

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI
from genkit_middleware import Fallback, Middleware, Retry

ai = Genkit(plugins=[GoogleAI(), Middleware()])


async def main() -> None:
    response = await ai.generate(
        model=GoogleAI.gemini('gemini-flash-latest'),
        prompt='Hello!',
        use=[
            Retry(max_retries=5),
            Fallback(models=['googleai/gemini-3.1-pro-preview']),
        ],
    )
    print(response.text)


if __name__ == '__main__':
    ai.run_main(main())
```

Every middleware works when you pass an instance in `use=[]`; no plugin
registration is needed. Registering `Middleware()` lists the stable middleware
in the Dev UI and lets a `.prompt` file name them in `use:`.

## Installation

```bash
uv add genkit-middleware genkit-google-genai
```

## Usage

### Retry

Retries a failed model call with exponential backoff. A `GenkitError` is retried only
when its status is in `statuses`; any other exception is retried:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Retry

retry = Retry(
    max_retries=3,
    statuses=['UNAVAILABLE', 'DEADLINE_EXCEEDED', 'RESOURCE_EXHAUSTED'],
    initial_delay_ms=1000,
    max_delay_ms=60000,
    backoff_factor=2.0,
    no_jitter=False,  # set True for deterministic backoff (tests)
)

response = await ai.generate(
    model=GoogleAI.gemini('gemini-flash-latest'),
    prompt='Hello!',
    use=[retry],
)
print(response.text)
# => Hello! How can I help you today?
```

### Fallback

Tries each backup model in order when the primary fails with a listed `GenkitError`
status (defaults: the transient statuses plus `NOT_FOUND` and `UNIMPLEMENTED`). Other
exceptions are raised as-is. Order matters: `use=[Retry(...), Fallback(...)]` retries
the whole sequence; `use=[Fallback(...), Retry(...)]` retries only the primary:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Fallback

fallback = Fallback(
    models=[
        'googleai/gemini-3.1-pro-preview',
        'googleai/gemini-flash-latest',
    ],
    statuses=['UNAVAILABLE', 'DEADLINE_EXCEEDED'],
)

response = await ai.generate(
    model=GoogleAI.gemini('gemini-pro-latest'),
    prompt='Hello!',
    use=[fallback],
)
print(response.text)
# => Hello! How can I help you today?
```

### ToolApproval

Requires approval before executing tools (useful for sensitive operations):

```python
from pydantic import BaseModel, Field

from genkit_google_genai import GoogleAI
from genkit_middleware import ToolApproval


class DeleteInput(BaseModel):
    name: str = Field(description='Database name to delete')


@ai.tool
async def delete_database(input: DeleteInput) -> str:
    return f'Deleted {input.name}'


approval = ToolApproval(
    allowed_tools=['get_weather', 'search'],  # These tools run without approval
)

first = await ai.generate(
    model=GoogleAI.gemini('gemini-flash-latest'),
    prompt='Delete the database',
    tools=['delete_database'],
    use=[approval],
)
```

When a non-allowed tool is called, execution is interrupted. Approve and re-run the
tool by restarting it with `resumed_metadata` that includes `tool_approved`. Pass the
history in `messages=` and leave out `prompt=`: a new prompt appends a user turn after
the pending tool request, so the call fails before the model or the tool runs. It
doesn't raise; it returns `finish_reason='failed'` with `response.error.status ==
'FAILED_PRECONDITION'` and `response.error.reason == 'INVALID_RESUME'`.

```python
response = await ai.generate(
    model=GoogleAI.gemini('gemini-flash-latest'),
    messages=list(first.messages),
    tools=['delete_database'],
    use=[approval],
    resume_restart=first.interrupts[0].restart(resumed_metadata={'tool_approved': True}),
)
print(response.text)
# => The database has been deleted.
```

Inside the tool, `ctx.is_resumed()` is `True` on the approved re-run and
`ctx.resumed_metadata` holds the dict you passed (`{'tool_approved': True}`).

### Skills

Scans directories for SKILL.md files and exposes them as loadable instructions:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Skills

skills = Skills(
    skill_paths=['skills', 'prompts/skills'],
)

response = await ai.generate(
    model=GoogleAI.gemini('gemini-flash-latest'),
    prompt='Help me with Python',
    use=[skills],
)
print(response.text)
# => Sure. Loaded python-expert. What are you working on?
```

Each direct subdirectory of a skill path that holds a `SKILL.md` is a skill, named after the directory (`skills/python-expert/SKILL.md` is `python-expert`). Frontmatter `description` shows up in the catalog; frontmatter `name` doesn't rename the skill:

```markdown
---
name: python-expert
description: Expert Python programming assistance
---

You are an expert Python programmer...
```

When two skill paths hold the same name, the later path wins. A symlinked skill directory is followed, so `skills/plating -> ~/shared/plating` works. A `SKILL.md` that is itself a symlink, not a regular file, or over 1 MiB isn't read.

### Filesystem

Provides sandboxed file operations confined to a root directory:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Filesystem

fs = Filesystem(
    root_dir='./workspace',
    allow_write_access=True,
    tool_name_prefix='',
)

response = await ai.generate(
    model=GoogleAI.gemini('gemini-flash-latest'),
    prompt='List files in the current directory',
    use=[fs],
)
print(response.text)
# => The workspace has menu.md and orders.csv.
```

Provides four tools:
- `list_files`: List files in a directory
- `read_file`: Read file content
- `write_file`: Write to a file (requires `allow_write_access=True`)
- `edit_file`: Edit file with string replacements (requires `allow_write_access=True`)

Paths resolve against `root_dir`. An absolute path works only if it already points inside the root; `/etc/hosts` is refused, not read as `{root}/etc/hosts`. Symlinks are resolved first, so a link out of the root is refused and left out of `list_files`. Tool errors come back as the tool response with the host root path removed.

### Artifacts

Exposes `read_artifact` / `write_artifact` tools and lists session artifacts in the
system prompt. It only works inside an agent, so it's experimental and lives in
`genkit_middleware.exp`. Pass it straight to the agent; `Middleware()` doesn't
register it, so it isn't listed in the Dev UI and a `.prompt` file can't name it:

```python
from genkit.exp import Genkit
from genkit.exp.agent import InMemorySessionStore
from genkit_google_genai import GoogleAI
from genkit_middleware import Middleware
from genkit_middleware.exp import Artifacts

ai = Genkit(plugins=[GoogleAI(), Middleware()])

agent = ai.define_agent(
    name='workspaceAgent',
    model=GoogleAI.gemini('gemini-flash-latest'),
    use=[Artifacts()],
    store=InMemorySessionStore(),
)

chat = agent.chat()
await chat.send('Write poem.txt with a short poem about Python agents.')
print([a.name for a in chat.artifacts])
# => ['poem.txt']
```
