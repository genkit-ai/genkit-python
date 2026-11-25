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
points it at a proxy or gateway in front of OpenAI.
``Mistral()`` (or ``OpenAI(provider='mistral')``) registers Mistral's chat
models and embedders under ``mistral/``. ``OpenAICompatible(name=..., base_url=...)``
registers any other OpenAI-compatible server (OpenRouter, Together, Groq,
vLLM) under the name you give it, with no OpenAI catalog.
``OpenAICompatible(backend=GoogleCloud())`` serves Model Garden's
OpenAI-compatible models (Llama, Mistral and others) under ``modelgarden/``
(``uv add 'genkit-openai[google-cloud]'``).

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
    The plugin key comes from ``api_key=`` or ``OPENAI_API_KEY``
    (``MISTRAL_API_KEY`` when the target is Mistral). A
    ``generate()`` or ``embed()`` call can bring its own key as
    ``context={'secrets': {'api_key': ...}}``; a plugin with no key of its
    own serves only those calls.

See Also:
    - OpenAI documentation: https://platform.openai.com/docs/
"""

from ._compatible import OpenAICompatible
from ._google_cloud import GoogleCloud
from ._openai_plugin import Mistral, OpenAI
from ._typing import OpenAIConfig

__all__ = ['GoogleCloud', 'Mistral', 'OpenAI', 'OpenAICompatible', 'OpenAIConfig']
