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

"""Deprecated: ``genkit-vertexai`` is now ``genkit-model-garden``.

This package re-exports ``ModelGarden`` from ``genkit_model_garden`` and warns
on import. It gets no releases for Genkit 1.0.

Example:
    ```python
    # 1. Replace the dependency, keeping the same extras
    #    uv remove genkit-vertexai
    #    uv add 'genkit-model-garden[anthropic]'

    # 2. Change the import; the plugin and model names are unchanged
    from genkit import Genkit
    from genkit_model_garden import ModelGarden

    ai = Genkit(plugins=[ModelGarden(project='bistro-prod', location='us-east5')])
    ```
"""

import warnings

from genkit_model_garden import ModelGarden

# FutureWarning rather than DeprecationWarning: Python hides DeprecationWarning
# unless it's attributed to __main__, and an import-time warning is attributed
# to the import machinery, so app developers would never see it.
warnings.warn(
    'genkit-vertexai is now genkit-model-garden. Replace the dependency with genkit-model-garden '
    '(same [anthropic] and [openai] extras) and import ModelGarden from genkit_model_garden. '
    'genkit-vertexai gets no releases for Genkit 1.0.',
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    'ModelGarden',
]
