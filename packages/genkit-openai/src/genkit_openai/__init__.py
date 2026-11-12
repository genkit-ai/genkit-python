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


"""OpenAI-compatible model provider for Genkit.

Registers OpenAI chat, image, text-to-speech, transcription, and embedding
models under ``openai/``, using the official OpenAI Python SDK. ``base_url=``
points it at any OpenAI-compatible server, such as Azure OpenAI or Together AI.

Example:
    ```python
    from genkit import Genkit
    from genkit_openai import OpenAI

    # 1. Initialize Genkit with the OpenAI plugin
    ai = Genkit(plugins=[OpenAI()])


    async def main() -> None:
        # 2. Generate with GPT-5.2
        res = await ai.generate(
            model=OpenAI.gpt('gpt-5.2'),
            prompt='Suggest 2 catchy names for an AI newsletter.',
        )

        # 3. Read the text
        print(res.text)
        # => 1. The Neural Brief
        #    2. Prompt & Circumstance


    if __name__ == '__main__':
        ai.run_main(main())
    ```

Credentials:
    The plugin key comes from ``api_key=`` or ``OPENAI_API_KEY``. A
    ``generate()`` or ``embed()`` call can bring its own key as
    ``context={'secrets': {'api_key': ...}}``; a plugin with no key of its
    own serves only those calls.

See Also:
    - OpenAI documentation: https://platform.openai.com/docs/
"""

from ._openai_plugin import OpenAI
from ._typing import OpenAIConfig

__all__ = ['OpenAI', 'OpenAIConfig']
