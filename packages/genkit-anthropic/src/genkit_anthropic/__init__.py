# Copyright 2025 Google LLC
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

"""Anthropic plugin for Genkit.

Registers Claude models as Genkit model actions under ``anthropic/``. Any
Claude id resolves on demand; ``Anthropic.claude()`` returns a typed ref bound
to ``AnthropicConfig``.

Example:
    ```python
    from genkit import Genkit
    from genkit_anthropic import Anthropic

    # 1. Initialize Genkit with the Anthropic plugin
    ai = Genkit(plugins=[Anthropic()])


    async def main() -> None:
        # 2. Generate with Claude Sonnet 4.6
        res = await ai.generate(
            model=Anthropic.claude('claude-sonnet-4-6'),
            prompt='Explain recursion in 10 words.',
        )

        # 3. Read the text
        print(res.text)
        # => A function that solves a problem by calling itself repeatedly.


    if __name__ == '__main__':
        ai.run_main(main())
    ```

Credentials:
    The plugin key comes from ``api_key=`` or ``ANTHROPIC_API_KEY``. A call
    can bring its own key as ``context={'secrets': {'api_key': ...}}``; a
    plugin with no key of its own serves only those calls.

See Also:
    - Anthropic documentation: https://docs.anthropic.com/
"""

from genkit_anthropic._config import AnthropicConfig, Effort, ThinkingDisplay, ThinkingMode
from genkit_anthropic._plugin import Anthropic

__all__ = [
    'Anthropic',
    'AnthropicConfig',
    'Effort',
    'ThinkingDisplay',
    'ThinkingMode',
]
