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

"""The genkit-vertexai shim re-exports ModelGarden and tells users to switch."""

import importlib
import sys
from types import ModuleType

import genkit_model_garden
import pytest


def _fresh_import() -> ModuleType:
    sys.modules.pop('genkit_vertexai', None)
    return importlib.import_module('genkit_vertexai')


def test_import_warns_with_replacement_package_and_module() -> None:
    with pytest.warns(FutureWarning) as record:
        _fresh_import()

    message = str(record[0].message)
    assert 'genkit-model-garden' in message
    assert 'genkit_model_garden' in message


def test_model_garden_is_the_same_class_as_genkit_model_garden() -> None:
    with pytest.warns(FutureWarning):
        shim = _fresh_import()

    assert shim.ModelGarden is genkit_model_garden.ModelGarden
    assert shim.__all__ == ['ModelGarden']
