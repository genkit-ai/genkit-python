# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A Genkit app's tools, served to MCP clients over stdio."""

import asyncio

from genkit_mcp import create_mcp_server
from pydantic import BaseModel

from genkit import Genkit, PublicError

ai = Genkit()

# Title: (topic, copies on the shelf).
BOOKS = {
    'The Hobbit': ('fantasy', 3),
    'A Wizard of Earthsea': ('fantasy', 0),
    'Dune': ('science fiction', 1),
}
HOURS = {'saturday': '9am to 6pm', 'sunday': 'closed'}


# MCP arguments are always an object, so every tool takes a model rather than a bare string.
class Day(BaseModel):
    name: str


class Topic(BaseModel):
    topic: str


class Title(BaseModel):
    title: str


class Results(BaseModel):
    titles: list[str]


class Stock(BaseModel):
    title: str
    copies: int


@ai.tool()
async def opening_hours(day: Day) -> str:
    """Report the shop's opening hours for a day of the week."""
    return HOURS.get(day.name.lower(), '10am to 7pm')


@ai.tool()
async def search_books(query: Topic) -> Results:
    """List the titles the shop carries on a topic."""
    return Results(titles=[title for title, (topic, _) in BOOKS.items() if topic == query.topic.lower()])


@ai.tool()
async def check_stock(book: Title) -> Stock:
    """Say how many copies of a title are on the shelf."""
    if book.title not in BOOKS:
        # A PublicError's message reaches the client; any other exception's text does not.
        raise PublicError('NOT_FOUND', f'We do not carry {book.title!r}')
    _, copies = BOOKS[book.title]
    return Stock(title=book.title, copies=copies)


server = create_mcp_server(ai, 'bookshop', instructions='Ask about books, stock and opening hours.')

if __name__ == '__main__':
    asyncio.run(server.start())
