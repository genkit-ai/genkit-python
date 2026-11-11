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

"""Experimental middleware that depends on agent sessions. May change in any release.

* ``Artifacts`` — ``read_artifact`` / ``write_artifact`` tools plus an artifact
  listing in the system prompt. Pass it straight to an agent:

    ```python
    from genkit.exp import Genkit
    from genkit.exp.agent import InMemorySessionStore
    from genkit_google_genai import GoogleAI
    from genkit_middleware import Middleware
    from genkit_middleware.exp import Artifacts

    # 1. Register the plugin and pass Artifacts straight to the agent
    ai = Genkit(plugins=[GoogleAI(), Middleware()])
    agent = ai.define_agent(
        name='workspaceAgent',
        model=GoogleAI.gemini('gemini-flash-latest'),
        use=[Artifacts()],
        store=InMemorySessionStore(),
    )

    # 2. The model writes an artifact through write_artifact
    chat = agent.chat()
    await chat.send('Write poem.txt with a short poem about Python agents.')
    print([a.name for a in chat.artifacts])
    # => ['poem.txt']
    ```

``Artifacts`` isn't registered by the ``Middleware()`` plugin, so it doesn't
show up in the Dev UI and a ``.prompt`` file can't name it in ``use:``.
"""

from genkit_middleware._artifacts import Artifacts

__all__ = ['Artifacts']
