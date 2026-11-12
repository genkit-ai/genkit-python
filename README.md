![Genkit logo](docs/resources/genkit-logo-dark.png#gh-dark-mode-only 'Genkit')
![Genkit logo](docs/resources/genkit-logo.png#gh-light-mode-only 'Genkit')

# Genkit Python SDK

Genkit is Google's open-source framework for building full-stack, AI-powered and agentic applications for any platform.

[Documentation](https://genkit.dev/docs/python/get-started/) | [Samples](samples/) | [Discord](https://discord.gg/qXt5zzQKpc) | [Report Issue](https://github.com/genkit-ai/genkit-python/issues)

---

## Quick Start & Core Patterns

Install the SDK and model provider:

```bash
uv add genkit genkit-google-genai genkit-google-cloud genkit-middleware
export GEMINI_API_KEY="your-api-key"
```

### 1. Streaming Generation

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])


async def main() -> None:
    # 1. Stream text as the model writes it
    stream = ai.generate_stream(
        model="googleai/gemini-flash-latest",
        prompt="Stream a 2-line poem about space.",
    )
    async for chunk in stream:
        print(chunk.text, end="")
    print()

    # 2. Await the final response for usage and finish reason
    response = await stream.response
    print(response.finish_reason)
    # => stop


if __name__ == "__main__":
    ai.run_main(main())
```

### 2. Streaming Tool Calling & Structured Output

```python
from pydantic import BaseModel, Field
from genkit import Genkit
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])


# 1. Define a tool with a Pydantic input
class WeatherInput(BaseModel):
    city: str = Field(description="Target city name")


@ai.tool(description="Get current weather for a location")
async def get_weather(input: WeatherInput) -> str:
    return f"Sunny, 72°F in {input.city}"


# 2. Define the structured output schema
class ActivityPlan(BaseModel):
    activities: list[str]
    outfit: str


async def main() -> None:
    # 3. Stream; Genkit runs the tool calls and parses the output
    stream = ai.generate_stream(
        model="googleai/gemini-flash-latest",
        prompt="Suggest activities for Seattle today.",
        tools=[get_weather],
        output_schema=ActivityPlan,
    )
    async for chunk in stream:
        if chunk.text:
            print(chunk.text, end="")
    print()

    # 4. Read the validated Pydantic object
    response = await stream.response
    print(response.output)
    # => activities=['Kayak on Lake Union', 'Discovery Park'] outfit='Light jacket'


if __name__ == "__main__":
    ai.run_main(main())
```

### 3. Tool Approval Middleware & Restarts

```python
from pydantic import BaseModel
from genkit.exp import Genkit
from genkit_google_genai import GoogleAI
from genkit_middleware import Middleware, ToolApproval

ai = Genkit(plugins=[GoogleAI(), Middleware()])
tool_approval = ToolApproval(allowed_tools=[])


# 1. A tool takes one input, so group the fields in a model
class Transfer(BaseModel):
    amount: float
    to_account: str


@ai.tool(description="Transfer money to an account")
async def transfer_money(transfer: Transfer) -> str:
    return f"Transferred ${transfer.amount} to {transfer.to_account}"


# 2. Every tool call pauses for approval (allowed_tools=[])
agent = ai.define_agent(
    name="bankingAgent",
    model="googleai/gemini-flash-latest",
    system="Banking assistant. Call transfer_money when requested.",
    tools=[transfer_money],
    use=[tool_approval],
)


async def main() -> None:
    # 3. The turn stops at the pending transfer
    chat = agent.chat()
    out1 = await chat.send("Transfer $100 to account 999.")
    print(out1.finish_reason)
    # => interrupted

    # 4. Approve the pending calls and resume
    restarts = [intr.restart(resumed_metadata={"tool_approved": True}) for intr in out1.interrupts]
    out2 = await chat.resume(restart=restarts)
    print(out2.text)
    # => $100 has been transferred to account 999.


if __name__ == "__main__":
    ai.run_main(main())
```

### 4. Agent Loops & Persistent Sessions

Agents are experimental, and `FirestoreSessionStore` needs the `firestore` extra:

```bash
uv add 'genkit-google-cloud[firestore]'
```

```python
from genkit.exp import Genkit
from genkit_google_cloud.exp import FirestoreSessionStore
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

# 1. Persist multi-turn session history in Cloud Firestore
store = FirestoreSessionStore()

# 2. Define an agent backed by the store
agent = ai.define_agent(
    name="supportAgent",
    model="googleai/gemini-flash-latest",
    system="You are a helpful customer support agent.",
    store=store,
)


async def main() -> None:
    # 3. Each turn is saved; the next one sees the history
    chat = agent.chat()
    await chat.send("Hi, my name is Alex.")
    res = await chat.send("What was my name again?")
    print(res.text)
    # => Your name is Alex.


if __name__ == "__main__":
    ai.run_main(main())
```

---

## Key Capabilities

<table>
  <tr>
    <td><strong>Type-Safe by Design</strong></td>
    <td>Native Python type annotations and Pydantic models drive structured inputs, outputs, and tool schemas.</td>
  </tr>
  <tr>
    <td><strong>Unified Model API</strong></td>
    <td>Switch between Google Gemini, Anthropic Claude, OpenAI, Ollama, and Vertex AI with one interface.</td>
  </tr>
  <tr>
    <td><strong>Integrated Observability</strong></td>
    <td>Built-in OpenTelemetry tracing. Inspect execution graphs, token usage, latency, and step inputs/outputs locally in real-time.</td>
  </tr>
  <tr>
    <td><strong>Production Deployment</strong></td>
    <td>Expose flows as standard ASGI/WSGI applications compatible with FastAPI, Flask, Django, Cloud Run, or any serverless platform.</td>
  </tr>
</table>

---

## Developer Tools & Dev UI

Run flows, prompts and tools locally with the Genkit Developer UI and CLI.

```bash
genkit start -- uv run main.py
```

Key features:
- **Playground**: Run and experiment with Genkit flows, prompts, and tools in dedicated playgrounds.
- **Trace Inspector**: Analyze detailed execution traces, including step-by-step breakdowns of complex flows.

<img src="docs/resources/readme-ui-traces-screenshot.png" width="700" alt="Screenshot of Genkit Developer UI showing traces">

---

## Exploring Samples & Onboarding

Reference implementations live in [`samples/`](samples/):

- **[Basic Flows](samples/basic-flows)**: Typed flows, named `ai.run` steps, streaming, and serving over FastAPI. No model or API key needed
- **[Agents](samples/agents)**: Autonomous multi-turn agents, session state persistence, and memory
- **[Multimodal & Advanced Gemini](samples/google-genai-media)**: Multimodal media processing, context caching, and code execution
- **[Tool Interrupts](samples/tool-interrupts)**: Pausing tool calls for human approval and resuming the turn

### Running Samples Locally

Clone the repository and launch any sample with the interactive Dev UI:

```bash
cd samples/<sample-name>
genkit start -- uv run src/main.py
```

---

## Local Development

If you're contributing to the Python SDK:

1. **Prerequisites**: Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. **Install Dependencies**: `uv sync`
3. **Run Linters & Tests**: `just lint` and `just test`

For coding standards and detailed guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Connect with Us

- [**Follow us on X/Twitter**](https://x.com/GenkitFramework) – News, updates, and tips.
- [**Join us on Reddit**](https://reddit.com/r/GenkitFramework) – Community discussion and Q&A.
- [**Join us on Discord**](https://discord.gg/qXt5zzQKpc) – Get real-time help and chat with developers.
- [**Contribute on GitHub**](https://github.com/genkit-ai/genkit-python/issues) – Report bugs, suggest features, or submit PRs.

## License
Apache 2.0