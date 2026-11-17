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

"""Model Garden plugin for Genkit.

Calls third-party models hosted in Google Cloud Model Garden (Claude, Llama,
Mistral) on your Google Cloud project's credentials and billing. Gemini is not
in this package; it's ``VertexAI`` in ``genkit-google-genai``.

Example:
    ```python
    from genkit import Genkit
    from genkit_model_garden import ModelGarden

    # 1. Initialize Genkit with the Model Garden plugin
    ai = Genkit(
        plugins=[ModelGarden(project='my-project', location='us-central1')],
    )


    # 2. Call models under the modelgarden/ namespace (not vertexai/)
    async def main() -> None:
        res = await ai.generate(
            model='modelgarden/anthropic/claude-sonnet-4-6',
            prompt='Explain recursion in 10 words.',
        )

        # 3. Inspect output shapes directly
        print(res.text)
        # => A function calling itself until reaching a base stopping condition.


    if __name__ == '__main__':
        ai.run_main(main())
    ```

Requirements:
    - Uses Google Cloud Application Default Credentials (ADC); there is no
      explicit-credentials option.
    - Claude models need ``genkit-model-garden[anthropic]``; Llama, Mistral, and other
      OpenAI-compatible models need ``genkit-model-garden[openai]``.

See Also:
    - Model Garden: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-garden/explore-models
"""

from genkit_model_garden._plugin import ModelGarden

__all__ = [
    'ModelGarden',
]
