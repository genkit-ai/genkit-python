# Genkit MCP

Connects Genkit to a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio and
exposes that server's tools as Genkit `tool` actions through a dynamic action provider. It also
serves a Genkit app's own tools to MCP clients.

Tracking issues: <https://github.com/genkit-ai/genkit/issues/6194> (client),
<https://github.com/genkit-ai/genkit-python/issues/91> (server)

## Usage

The example uses Gemini through `genkit-google-genai` and needs `GEMINI_API_KEY` set. Any model plugin works.

```python
import asyncio

from genkit import Genkit
from genkit_google_genai import GoogleAI
from genkit_mcp import define_mcp_client

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))

client = define_mcp_client(
    ai,
    'everything',
    command='npx',
    args=['-y', '@modelcontextprotocol/server-everything'],
)


async def main() -> None:
    try:
        # What the server offers, under the names a model is shown.
        tools = await client.get_active_tools()
        for tool in tools:
            print(tool.name, tool.description)

        # Every tool of this server, or just one.
        result = await ai.generate(prompt='Echo "hello".', tools=['everything:tool/*'])
        one = f'{client.name}:tool/{client.tool_name("echo")}'
        result = await ai.generate(prompt='Echo "hello".', tools=[one])

        # The listing itself is a list of Genkit tools, so it can be handed to a
        # generate call or called directly.
        result = await ai.generate(prompt='Echo "hello".', tools=tools)
        print(result.text)
        echoed = (await tools[0]({'message': 'hello'})).output
        print(echoed)
    finally:
        await client.close()


asyncio.run(main())
```

## Tool names

Every tool is named `<prefix>_<tool>`, so the tool above is `everything_echo`. The prefix is the
provider name you registered the server under, `everything` here. The separator is an underscore
because Gemini and OpenAI reject a `/` in a function declaration name. A prefix holding a character
they reject fails when the client is defined, rather than on your first `generate`. A tool name the
server lists with such a character, such as `look.up`, has it rewritten to an underscore with a
warning. Two tools that then share a name, such as `look.up` and `look/up`, are both left out with a
warning, since either could answer a call meant for the other. Length is not checked at definition,
because the tool half of the name is unknown until the server lists it. A `<prefix>_<tool>` longer
than 64 characters is warned about when the tools are listed and passed through, so the provider's
400 arrives on `generate`. Nor are names checked across servers: a
prefix `fake` with a tool `await_stall` and a prefix `fake_await` with a tool `stall` both make
`fake_await_stall`, and `generate` refuses to bind both. Each prefix names one provider, so a second
client under a name already registered fails when it is defined, unless the client holding the name
is closed.

The prefix is deliberately not the name the server advertises for itself, which is arbitrary
server-controlled text: `@modelcontextprotocol/server-everything` above calls itself
`mcp-servers/everything`, and that slash would be a 400 from both providers.

## Connecting and disconnecting

Connecting is lazy. The child process starts the first time something lists the tools, once per
event loop, and the Dev UI's reflection server has a loop of its own. So under `genkit start`, a
flow run from the Dev UI and your `main()` talk to two separate server processes. A server that
keeps state, such as a session or an open file, keeps one copy of it for each, and servers that bind
a port, take a lock file or own a browser profile can clash.

`await client.restart()` reconnects a live client, for a server whose configuration or state changed
underneath you. A request still in flight when it restarts, on any event loop, fails with status
`UNAVAILABLE` and can be retried. `await client.close()` disconnects and stops the child process: a
closed client cannot be restarted. To use that server again, define a new client. Afterwards
`generate` with one of its tools, and `get_active_tools()`, fail with a `GenkitError` whose status
is `FAILED_PRECONDITION`. A server that cannot start, or whose process dies, fails with status
`UNAVAILABLE`, and a request the server does not answer within `request_timeout_millis` fails with
`DEADLINE_EXCEEDED`. A connection that died on its own is replaced on next use, so one bad server start is
not permanent. Resolving a tool by its action key, as the Dev UI does to run one, reports neither
status: core finds no action for a closed client or a server that cannot start.

Nothing needs closing at exit. Call `close()` when you are done with a server before your process
is, not as cleanup on the way out: a process running under `genkit start` outlives your `main()`,
and a closed client's tools can no longer be selected by a flow. The Dev UI lists each registered
client's tools, and none for a closed one or one whose server cannot start.

## Several servers behind one name

`define_mcp_host` registers one provider over any number of servers, so a single selector reaches
all of them:

```python
from genkit import Genkit
from genkit_mcp import define_mcp_host

ai = Genkit()

host = define_mcp_host(
    ai,
    'hosted',
    mcp_servers={
        'everything': {'command': 'npx', 'args': ['-y', '@modelcontextprotocol/server-everything']},
        'git': {'command': 'uvx', 'args': ['mcp-server-git']},
    },
)

result = await ai.generate(prompt='Echo "hello".', tools=['hosted:tool/*'])
one = f'{host.name}:tool/{host.tool_name("everything", "echo")}'
result = await ai.generate(prompt='Echo "hello".', tools=[one])
```

`mcp_servers` takes an `mcpServers` block of stdio servers, as Claude Desktop and Cursor write it, so
such a block can be loaded from its file: `mcp_servers=json.load(f)['mcpServers']`. An entry that is
not a stdio entry, such as one with a `url` or `type: 'sse'`, fails the host with an error naming
that server. So does an entry with any key besides `command`, `args`, `env`, `cwd`, `type` and
`disabled`. Some clients write `disabled: true` to switch a server off without deleting its entry.
Such an entry is checked the same way, but its server is never started or listed.
`await host.connect(name, entry)` takes one entry too, so a server from the same file can be added
later: `await host.connect('git', config['mcpServers']['git'])`. Connecting a disabled entry stops
any server of that name.

An entry's `env` often holds secrets, and a traceback that renders frame locals, as Genkit's console
logging does, can print them. Once a host is defined, the plugin keeps `env` out of its own frames.
It cannot do that for your frames: in the `connect` call above, `config` stays a local of the
calling function, so a failed connect logged with its traceback prints the whole block. The same
goes for a `mcp_servers` mapping that fails validation when the host is defined. The child does not
inherit this process's environment, so the secrets have to be in `env`. To keep them out of your
frames, build the entry in a helper and pass its result straight on, as in
`await host.connect('git', load_server('git'))`, so no frame that can still fail holds it.

Each server keeps its own namespace, so the tools above are `everything_echo` and `git_git_log`. The
key a server has in `mcp_servers` is its prefix. The host's own name is only the provider name, so
it has to fit a selector but no model ever reads it.

A server that cannot start, or whose process dies, fails with status `UNAVAILABLE`, and one that
does not answer a request in time fails with `DEADLINE_EXCEEDED`. The host logs either and skips it,
so the other servers stay selectable. A server disconnected or replaced while the host was listing
it is left out of that listing, and a server that does not advertise the tools capability is not
asked for tools and lists none. Any other failure fails the whole listing:
a server that breaks the protocol, a server that answers the listing with an error, and two servers
that serve a tool of the same name. Only the server you ask for is affected by `await host.connect(...)`,
`await host.disconnect(...)` and `await host.reconnect(...)`. `connect` and `reconnect` start the
server and list it before they return, so a server that cannot start fails right there. A server
that `connect` cannot list is stopped, and the host keeps what it held before, including any server
of that name. `await host.close()` disconnects every
server the host holds, and like `McpClient.close()` it is final: afterwards the host fails with
status `FAILED_PRECONDITION`, so define a new host to use those servers again.

One provider means one listing. A tool call that fails on one server drops the listing for all of
them, so the next `generate` lists every server again.

## Serving your app as an MCP server

`create_mcp_server` wraps a `Genkit` instance and registers nothing, so the tools it serves are the
ones the app already has:

```python
import asyncio

from genkit import Genkit
from genkit_mcp import create_mcp_server
from pydantic import BaseModel

HOURS = {'saturday': '9am to 6pm', 'sunday': 'closed'}

ai = Genkit()


class Day(BaseModel):
    name: str


@ai.tool()
async def opening_hours(day: Day) -> str:
    """Report the shop's opening hours for a day of the week."""
    return HOURS.get(day.name.lower(), '10am to 7pm')


server = create_mcp_server(ai, 'bookshop', instructions='Ask about the shop.')

if __name__ == '__main__':
    asyncio.run(server.start())
```

The app's tools are read with `ai.list_tools()` on every `tools/list`, so a tool defined after the
server was built is still served. A tool whose input is not an object, such as a single scalar or an
optional model, is skipped: MCP arguments are always an object, and such a tool is not callable by
name either, rather than being listed and then failing. A model that refers to itself is served.
An output schema is advertised only for a tool built as a `Tool`, as `@ai.tool()` builds one, that
returns an object itself, such as `-> Shelf`. A client rejects a result that lacks the structured
content a schema promised, and a tool returning `MultipartToolResponse[Shelf]` may leave its output
out. Tools a dynamic action provider supplies, from `define_mcp_client` for instance, are not
defined on the app and are not served.

Arguments that do not match a tool's input schema are rejected by the MCP SDK before the tool runs,
with an error that quotes them. A tool that raises answers with an error result, worded as a served
flow's error body is. A `PublicError` sends its status and message, and arguments the input model's
own validators reject send the failing fields without their values. Any other `GenkitError` sends
`Tool '<name>' failed: <status>`, and any other exception `Tool '<name>' failed`, since a provider's
error text can name a project, a resource path or a URL. So raise a `PublicError` for a failure the
model should read and retry. Such an error, like rejected arguments, is logged on the server as one
warning line. Any other failure is logged with its traceback. A tool that interrupts for the user's
input answers with an error result saying so, since MCP cannot pause a call. A result that cannot
be encoded as UTF-8 answers with `Tool '<name>' failed`. A plugin that
fails to start fails the listing, and the client is told only that the server could not list its
tools.

### stdio

`await server.start()` serves on this process's stdin and stdout until stdin closes. They are the
wire, so `start()` gives the transport copies of file descriptors 0 and 1, whatever `sys.stdin` and
`sys.stdout` are at the time. It then points descriptor 1 and `sys.stdout` at stderr, and descriptor
0 at the null device, for the rest of the process's life. One stray line corrupts a JSON-RPC frame,
and structlog's builtin default writes to stdout, so `print`, loggers, child processes and C
extensions all write to stderr from then on, and a child process cannot read the client's requests.
Text still buffered for stdout is flushed ahead of the first frame. Pass `redirect_stdout=False` to
leave both as they are, for an app that keeps them clear itself.

`await server.serve(read_stream, write_stream)` serves on any SDK stream pair instead, and touches
nothing.

### Known gaps

Scalar outputs and strings that look like JSON do not round-trip exactly. Clients join every text
block into the output, so a text or data content part is sent only beside no output or an object
output, and is dropped with a warning otherwise. A Genkit client reading a result with structured
content keeps that object and drops the text beside it. A media part whose URL is not absolute is
dropped with a warning, and one sent as a resource link is dropped by a Genkit client. A tool whose
input is not an object is not served at all, and an output schema is advertised only as described
above, so a client is told less about such a tool than Genkit knows. The advertised schema is the
one Genkit validates input against, so an output model whose dumped JSON differs from it, through a
`serialization_alias` or a `computed_field` on a model that forbids extra fields, fails the client's
check of the result. Nothing sends `notifications/tools/list_changed`; the list is re-read per
request instead. A client's `notifications/cancelled` cancels the tool call. Progress, sampling and
elicitation are not supported. A plugin-namespaced tool name carries a `/`, which the MCP SDK warns
about. stdio reads and writes UTF-8 whatever the locale, but is untested on Windows.
