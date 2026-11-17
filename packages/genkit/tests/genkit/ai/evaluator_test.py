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

"""Plugin authors build evaluators with Evaluator / BatchEvaluator and list them in list_actions()."""

import subprocess  # noqa: S404
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from genkit import Genkit, GenkitError
from genkit.evaluator import (
    BaseDataPoint,
    BatchEvaluator,
    Evaluator,
    EvaluatorRequest,
    EvaluatorResponse,
    Score,
)
from genkit.plugin_api import Action, ActionKind, Plugin


class ThresholdOptions(BaseModel):
    threshold: float = 0.5


async def _exact_match(datapoint: BaseDataPoint, options: object | None) -> EvaluatorResponse:
    return EvaluatorResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=[Score(score=datapoint.output == datapoint.reference)],
    )


def _two_rows() -> list[BaseDataPoint]:
    return [
        BaseDataPoint(input='2+2', output='4', reference='4', test_case_id='case1'),
        BaseDataPoint(input='3+3', output='7', reference='6', test_case_id='case2'),
    ]


class ExactMatchPlugin(Plugin):
    """Plugin that hands Genkit an evaluator built with Evaluator(...) on resolve."""

    name = 'p'

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.EVALUATOR or name != 'e':
            return None
        return Evaluator('p/e', _exact_match, display_name='Exact match', definition='output equals reference')

    async def list_actions(self) -> list[Action]:
        return [Evaluator('p/e', _exact_match, display_name='Exact match', definition='output equals reference')]


@pytest.mark.asyncio
async def test_evaluator_builds_action_without_registering() -> None:
    """Evaluator('t/e', fn, ...) is an EVALUATOR action; an app with no plugin returning it can't evaluate with it."""
    ai = Genkit()

    action = Evaluator('t/e', _exact_match, display_name='Exact match', definition='output equals reference')

    assert isinstance(action, Action)
    assert action.kind == ActionKind.EVALUATOR
    assert action.name == 't/e'
    with pytest.raises(GenkitError) as raised:
        await ai.evaluate(evaluator='t/e', dataset=_two_rows())
    assert raised.value.status == 'NOT_FOUND'


@pytest.mark.asyncio
async def test_evaluator_action_runs_fn_per_datapoint() -> None:
    """Running the built action over two rows returns two scored rows, in dataset order."""
    action = Evaluator('t/e', _exact_match, display_name='Exact match', definition='output equals reference')

    result = await action.run(EvaluatorRequest(dataset=_two_rows(), eval_run_id='run-1'))

    rows = result.response.root
    assert [row.test_case_id for row in rows] == ['case1', 'case2']
    assert [score.score for score in rows[0].evaluation] == [True]
    assert [score.score for score in rows[1].evaluation] == [False]


def test_evaluator_sync_fn_raises_type_error() -> None:
    """Evaluator(...) rejects a sync fn when built, instead of failing every row at run time."""

    def sync_match(datapoint: BaseDataPoint, options: object | None) -> EvaluatorResponse:
        return EvaluatorResponse(test_case_id=datapoint.test_case_id or '', evaluation=[Score(score=True)])

    with pytest.raises(TypeError, match="Got sync function for 't/e'"):
        Evaluator('t/e', cast(Any, sync_match), display_name='Exact match', definition='output equals reference')


def test_evaluator_action_metadata_carries_display_name_definition_billed() -> None:
    """The listing for an evaluator carries the display name, definition, billed flag and options the Dev UI shows."""
    listing = Evaluator(
        'p/fluency',
        _exact_match,
        display_name='Fluency',
        definition='Assesses the language mastery of an output',
        is_billed=True,
        config_schema=ThresholdOptions,
    )._action_metadata()

    assert listing.action_type == ActionKind.EVALUATOR
    assert listing.name == 'p/fluency'
    assert listing.metadata == {
        'evaluator': {
            'evaluatorDisplayName': 'Fluency',
            'evaluatorDefinition': 'Assesses the language mastery of an output',
            'evaluatorIsBilled': True,
            'label': 'p/fluency',
            'customOptions': {
                'properties': {'threshold': {'default': 0.5, 'title': 'Threshold', 'type': 'number'}},
                'title': 'ThresholdOptions',
                'type': 'object',
            },
        }
    }


@pytest.mark.parametrize('config_schema', [None, ThresholdOptions], ids=['no_options', 'with_options'])
def test_evaluator_and_metadata_agree_for_same_args(config_schema: type[BaseModel] | None) -> None:
    """The built evaluator and its listing show the Dev UI the same card for the same args."""
    action = Evaluator(
        'p/e',
        _exact_match,
        display_name='Exact match',
        definition='output equals reference',
        is_billed=True,
        config_schema=config_schema,
    )
    listing = action._action_metadata()

    assert listing.metadata is not None
    assert action.metadata['evaluator'] == listing.metadata['evaluator']
    assert listing.input_schema == action.input_schema
    assert listing.output_schema == action.output_schema


@pytest.mark.asyncio
async def test_plugin_returning_evaluator_registers_it_on_app() -> None:
    """A plugin whose resolve() returns Evaluator(...) makes ai.evaluate(evaluator='p/e', ...) score every row."""
    ai = Genkit(plugins=[ExactMatchPlugin()])

    results = await ai.evaluate(evaluator='p/e', dataset=_two_rows())

    assert [row.test_case_id for row in results] == ['case1', 'case2']
    assert [score.score for score in results[0].evaluation] == [True]
    assert [score.score for score in results[1].evaluation] == [False]


async def _exact_match_batch(req: EvaluatorRequest) -> list[EvaluatorResponse]:
    return [await _exact_match(datapoint, None) for datapoint in req.dataset]


class ExactMatchBatchPlugin(Plugin):
    """Plugin that hands Genkit a batch evaluator built with BatchEvaluator(...) from init."""

    name = 'b'

    async def init(self) -> list[Action]:
        return [
            BatchEvaluator('b/e', _exact_match_batch, display_name='Exact match', definition='output equals reference')
        ]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[Action]:
        return []


@pytest.mark.asyncio
async def test_batch_evaluator_builds_action_without_registering() -> None:
    """BatchEvaluator('t/b', fn, ...) is an EVALUATOR action that no app knows about until something registers it."""
    ai = Genkit()

    action = BatchEvaluator('t/b', _exact_match_batch, display_name='Exact match', definition='output equals reference')

    assert isinstance(action, Action)
    assert action.kind == ActionKind.EVALUATOR
    with pytest.raises(GenkitError) as raised:
        await ai.evaluate(evaluator='t/b', dataset=_two_rows())
    assert raised.value.status == 'NOT_FOUND'


def test_batch_evaluator_sync_fn_raises_type_error() -> None:
    """BatchEvaluator(...) rejects a sync fn when built."""

    def sync_batch(req: EvaluatorRequest) -> list[EvaluatorResponse]:
        return []

    with pytest.raises(TypeError, match="Got sync function for 't/b'"):
        BatchEvaluator('t/b', cast(Any, sync_batch), display_name='Exact match', definition='output equals reference')


def test_batch_evaluator_and_metadata_agree_for_same_args() -> None:
    """A built batch evaluator and its listing show the Dev UI the same card."""
    action = BatchEvaluator(
        'p/b', _exact_match_batch, display_name='Exact match', definition='output equals reference', is_billed=True
    )
    listing = action._action_metadata()

    assert listing.metadata is not None
    assert action.metadata['evaluator'] == listing.metadata['evaluator']
    assert listing.input_schema == action.input_schema
    assert listing.output_schema == action.output_schema


@pytest.mark.asyncio
async def test_plugin_returning_batch_evaluator_registers_it_on_app() -> None:
    """A plugin whose init() returns BatchEvaluator(...) makes ai.evaluate(evaluator='b/e', ...) score every row."""
    ai = Genkit(plugins=[ExactMatchBatchPlugin()])

    results = await ai.evaluate(evaluator='b/e', dataset=_two_rows())

    assert [row.test_case_id for row in results] == ['case1', 'case2']
    assert [score.score for score in results[0].evaluation] == [True]
    assert [score.score for score in results[1].evaluation] == [False]


@pytest.mark.asyncio
async def test_evaluator_surfaces_emit_single_future_warning_at_caller() -> None:
    """Defining or running evaluators warns once per call at the caller's line; plugin construction stays silent."""
    ai = Genkit()

    with warnings.catch_warnings(record=True) as build_caught:
        warnings.simplefilter('always')
        per_row = Evaluator('t/e', _exact_match, display_name='Exact match', definition='output equals reference')
        batch = BatchEvaluator(
            't/b', _exact_match_batch, display_name='Exact match', definition='output equals reference'
        )
    assert build_caught == []

    with warnings.catch_warnings(record=True) as def_caught:
        warnings.simplefilter('always')
        ai.define_evaluator('custom/e', _exact_match, display_name='Exact match', definition='output equals reference')
    assert len(def_caught) == 1
    assert def_caught[0].category is FutureWarning
    assert def_caught[0].filename == __file__
    assert 'deprecated in 0.14' in str(def_caught[0].message)

    with warnings.catch_warnings(record=True) as batch_def_caught:
        warnings.simplefilter('always')
        ai.define_batch_evaluator(
            'custom/b', _exact_match_batch, display_name='Exact match', definition='output equals reference'
        )
    assert len(batch_def_caught) == 1
    assert batch_def_caught[0].category is FutureWarning
    assert batch_def_caught[0].filename == __file__

    with warnings.catch_warnings(record=True) as eval_caught:
        warnings.simplefilter('always')
        await ai.evaluate(evaluator='custom/e', dataset=_two_rows())
    assert len(eval_caught) == 1
    assert eval_caught[0].category is FutureWarning
    assert eval_caught[0].filename == __file__

    with warnings.catch_warnings(record=True) as run_caught:
        warnings.simplefilter('always')
        await per_row.run(EvaluatorRequest(dataset=_two_rows(), eval_run_id='run-1'))
        await batch.run(EvaluatorRequest(dataset=_two_rows(), eval_run_id='run-2'))
    assert len(run_caught) == 2
    assert all(w.category is FutureWarning and w.filename == __file__ for w in run_caught)


def test_evaluator_warning_visible_by_default_outside_main(tmp_path: Path) -> None:
    """A non-__main__ module still prints the warning under default Python filters (no -W flags)."""
    mod_file = tmp_path / 'tutorial_mod.py'
    mod_file.write_text(
        'from genkit import Genkit\n'
        'from genkit.evaluator import BaseDataPoint, EvaluatorResponse, Score\n'
        'async def fn(dp: BaseDataPoint, _opts: object = None) -> EvaluatorResponse:\n'
        "    return EvaluatorResponse(test_case_id='', evaluation=[Score(score=True)])\n"
        'def setup() -> None:\n'
        '    ai = Genkit()\n'
        "    ai.define_evaluator('t/e', fn, display_name='T', definition='D')\n",
        encoding='utf-8',
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, '-c', 'import tutorial_mod; tutorial_mod.setup()'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = 'FutureWarning: Genkit evaluators are deprecated in 0.14 and will be removed in the next release.'
    assert expected in proc.stderr
    assert 'tutorial_mod.py:7' in proc.stderr
