# MCP server hello

A Genkit app's tools, served to any MCP client over stdio.

`src/main.py` is a bookshop app with three tools: `opening_hours`,
`search_books` and `check_stock`. It defines no model, so nothing here needs an
API key. `src/client.py` launches it and uses it with the MCP SDK's own
client, the way Claude Desktop, an IDE or another agent would.

```bash
uv sync
uv run src/client.py
```

```
Connected to bookshop: Ask about books, stock and opening hours.
Tools:
  check_stock: Say how many copies of a title are on the shelf.
  opening_hours: Report the shop's opening hours for a day of the week.
  search_books: List the titles the shop carries on a topic.
opening_hours({'name': 'Sunday'}) [ok]: closed
search_books({'topic': 'fantasy'}) [ok]: {'titles': ['The Hobbit', 'A Wizard of Earthsea']}
check_stock({'title': 'Dune'}) [ok]: {'title': 'Dune', 'copies': 1}
check_stock({'title': 'Ulysses'}) [error]: NOT_FOUND: We do not carry 'Ulysses'
```

To use it from another MCP client, give the client this command, run from this
folder:

```bash
uv run src/main.py
```

## What it shows

`create_mcp_server(ai, 'bookshop')` wraps the app and registers nothing. The
tools it serves are the ones defined with `@ai.tool()`, read again on every
`tools/list`. `server.start()` serves on stdin and stdout until the client
closes stdin.

MCP arguments are always an object, so each tool takes a Pydantic model and a
client sends that model's fields: `{'title': 'Dune'}`, not
`{'book': {'title': 'Dune'}}`. A tool that takes a bare `str` is not served.

`search_books` and `check_stock` return models, so they advertise an output
schema and answer with structured content. `opening_hours` returns a string,
which arrives as text.

`check_stock` raises a `PublicError` for a title the shop does not carry. The
client gets an error result with its status and message. Any other
`GenkitError` sends only `Tool 'check_stock' failed: <status>`, and any other
exception `Tool 'check_stock' failed`, since their text can carry a URL or a
credential.

stdout is the wire, so the server logs to stderr. The client passes the
server's stderr through, which is why the failed `check_stock` call prints a
traceback above the client's output.

## Tests

The tests launch the real server and drive it with the MCP SDK client:

```bash
uv run pytest samples/mcp-server-hello --no-cov   # from the repo root
```
