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

"""Tests for Interactions API converters."""

from __future__ import annotations

from typing import Any, cast

import pytest
from genkit_google_genai._interactions._converters import (
    ensure_tool_ids,
    from_interaction,
    from_interaction_content,
    from_interaction_step,
    from_interaction_sync,
    from_thought_step,
    parts_from_steps,
    split_system_instruction,
    to_interaction_content,
    to_interaction_role,
    to_interaction_steps,
    to_interaction_tool,
    to_thought_step,
    usage_from_interaction,
    with_basic_usage_stats,
)
from google.genai.interactions import Content, Interaction, Step, ThoughtStep, Usage
from pydantic import BaseModel, TypeAdapter

from genkit import GenkitError, Media, Message, Part
from genkit.model import ToolDefinition, ToolRequest, ToolResponse

ContentAdapter: TypeAdapter[Content] = TypeAdapter(Content)
StepAdapter: TypeAdapter[Step] = TypeAdapter(Step)


def part_dict(part: Part) -> dict:
    return part.model_dump(by_alias=True, exclude_none=True)


class TestEnsureToolIds:
    def test_assigns_ids_to_tool_requests_without_refs(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(tool_request=ToolRequest(name='tool1', input={})),
                    Part(tool_request=ToolRequest(name='tool2', input={})),
                ],
            )
        ]
        result = ensure_tool_ids(messages)
        req1 = result[0].content[0].tool_request
        req2 = result[0].content[1].tool_request
        assert req1 is not None and req1.ref and req1.ref.startswith('genkit-auto-id-')
        assert req2 is not None and req2.ref and req2.ref.startswith('genkit-auto-id-')
        assert req1.ref != req2.ref

    def test_assigns_matching_ids_to_tool_responses(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(tool_request=ToolRequest(name='tool1', input={})),
                    Part(tool_request=ToolRequest(name='tool2', input={})),
                ],
            ),
            Message(
                role='tool',
                content=[
                    Part(tool_response=ToolResponse(name='tool1', output={})),
                    Part(tool_response=ToolResponse(name='tool2', output={})),
                ],
            ),
        ]
        result = ensure_tool_ids(messages)
        req1 = result[0].content[0].tool_request
        req2 = result[0].content[1].tool_request
        res1 = result[1].content[0].tool_response
        res2 = result[1].content[1].tool_response
        assert req1 and req1.ref and res1 and res1.ref == req1.ref
        assert req2 and req2.ref and res2 and res2.ref == req2.ref

    def test_assigns_orphan_id_without_matching_request(self) -> None:
        messages = [
            Message(
                role='tool',
                content=[Part(tool_response=ToolResponse(name='tool1', output={}))],
            )
        ]
        result = ensure_tool_ids(messages)
        res1 = result[0].content[0].tool_response
        assert res1 and res1.ref and res1.ref.startswith('genkit-orphan-id-')

    def test_preserves_existing_refs(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(tool_request=ToolRequest(name='tool1', input={}, ref='existing-id'))],
            )
        ]
        result = ensure_tool_ids(messages)
        req1 = result[0].content[0].tool_request
        assert req1 and req1.ref == 'existing-id'

    def test_pairs_unreffed_responses_to_existing_request_refs(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(tool_request=ToolRequest(name='a', input={}, ref='existing')),
                    Part(tool_request=ToolRequest(name='b', input={})),
                ],
            ),
            Message(
                role='tool',
                content=[
                    Part(tool_response=ToolResponse(name='a', output={})),
                    Part(tool_response=ToolResponse(name='b', output={})),
                ],
            ),
        ]
        result = ensure_tool_ids(messages)
        req_a = result[0].content[0].tool_request
        req_b = result[0].content[1].tool_request
        res_a = result[1].content[0].tool_response
        res_b = result[1].content[1].tool_response
        assert req_a and req_a.ref == 'existing'
        assert req_b and req_b.ref and req_b.ref.startswith('genkit-auto-id-')
        assert res_a and res_a.ref == 'existing'
        assert res_b and res_b.ref == req_b.ref

    def test_skips_claimed_request_ids_when_pairing(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(tool_request=ToolRequest(name='a', input={}, ref='existing')),
                    Part(tool_request=ToolRequest(name='b', input={})),
                ],
            ),
            Message(
                role='tool',
                content=[
                    Part(tool_response=ToolResponse(name='a', output={}, ref='existing')),
                    Part(tool_response=ToolResponse(name='b', output={})),
                ],
            ),
        ]
        result = ensure_tool_ids(messages)
        req_b = result[0].content[1].tool_request
        res_b = result[1].content[1].tool_response
        assert req_b and res_b and res_b.ref == req_b.ref


class TestToInteractionRole:
    def test_user(self) -> None:
        assert to_interaction_role('user') == 'user'

    def test_model(self) -> None:
        assert to_interaction_role('model') == 'model'

    def test_tool_maps_to_user(self) -> None:
        assert to_interaction_role('tool') == 'user'

    def test_system_raises(self) -> None:
        with pytest.raises(GenkitError, match='system_instruction') as raised:
            to_interaction_role('system')
        assert raised.value.status == 'INVALID_ARGUMENT'


class TestToInteractionTool:
    def test_converts_tool_definition(self) -> None:
        tool = ToolDefinition(
            name='myFunc',
            description='desc',
            input_schema={'type': 'object', 'properties': {'arg': {'type': 'string'}}},
        )
        result = to_interaction_tool(tool)
        assert result == {
            'type': 'function',
            'name': 'myFunc',
            'description': 'desc',
            'parameters': {'type': 'object', 'properties': {'arg': {'type': 'string'}}},
        }


class TestToInteractionContent:
    def test_text(self) -> None:
        result = to_interaction_content(Part.from_text('Hello'))
        assert result == {'type': 'text', 'text': 'Hello'}

    def test_image_data(self) -> None:
        result = to_interaction_content(Part.from_media('data:image/png;base64,DATA', content_type='image/png'))
        assert result == {'type': 'image', 'data': 'DATA', 'mime_type': 'image/png'}

    def test_image_data_missing_separator(self) -> None:
        with pytest.raises(GenkitError, match='missing payload separator') as raised:
            to_interaction_content(Part.from_media('data:image/png;base64GARBAGE', content_type='image/png'))
        assert raised.value.status == 'INVALID_ARGUMENT'

    def test_image_uri(self) -> None:
        result = to_interaction_content(Part.from_media('gs://bucket/image.png', content_type='image/png'))
        assert result == {'type': 'image', 'uri': 'gs://bucket/image.png', 'mime_type': 'image/png'}

    def test_audio(self) -> None:
        result = to_interaction_content(Part.from_media('data:audio/mp3;base64,DATA', content_type='audio/mp3'))
        assert result == {'type': 'audio', 'data': 'DATA', 'mime_type': 'audio/mp3'}

    def test_document(self) -> None:
        result = to_interaction_content(Part.from_media('gs://bucket/doc.pdf', content_type='application/pdf'))
        assert result == {'type': 'document', 'uri': 'gs://bucket/doc.pdf', 'mime_type': 'application/pdf'}

    def test_csv_is_a_document(self) -> None:
        result = to_interaction_content(Part.from_media('data:text/csv;base64,DATA', content_type='text/csv'))
        assert result == {'type': 'document', 'data': 'DATA', 'mime_type': 'text/csv'}

    def test_word_and_excel_are_documents(self) -> None:
        # User report (JS 1.43): these threw "Unsupported media type", although
        # the server accepts them.
        docx = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        xlsx = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        assert to_interaction_content(Part.from_media(f'data:{docx};base64,DATA', content_type=docx)) == {
            'type': 'document',
            'data': 'DATA',
            'mime_type': docx,
        }
        assert to_interaction_content(Part.from_media('gs://bucket/sheet.xlsx', content_type=xlsx)) == {
            'type': 'document',
            'uri': 'gs://bucket/sheet.xlsx',
            'mime_type': xlsx,
        }

    def test_other_content_types_are_left_for_the_server_to_accept_or_reject(self) -> None:
        result = to_interaction_content(Part.from_media('gs://bucket/notes.txt', content_type='text/plain'))
        assert result == {'type': 'document', 'uri': 'gs://bucket/notes.txt', 'mime_type': 'text/plain'}

    def test_youtube_url_without_content_type_is_video_without_mime_type(self) -> None:
        url = 'https://www.youtube.com/watch?v=abc123'
        assert to_interaction_content(Part(media=Media(url=url))) == {'type': 'video', 'uri': url}
        assert to_interaction_content(Part(media=Media(url='https://youtu.be/abc123'))) == {
            'type': 'video',
            'uri': 'https://youtu.be/abc123',
        }

    def test_other_uri_without_known_content_type_is_document_without_mime_type(self) -> None:
        url = 'https://generativelanguage.googleapis.com/v1beta/files/abc123'
        assert to_interaction_content(Part(media=Media(url=url))) == {'type': 'document', 'uri': url}

    def test_content_type_from_a_known_file_extension(self) -> None:
        assert to_interaction_content(Part(media=Media(url='gs://bucket/doc.pdf'))) == {
            'type': 'document',
            'uri': 'gs://bucket/doc.pdf',
            'mime_type': 'application/pdf',
        }

    def test_content_type_from_a_data_url(self) -> None:
        assert to_interaction_content(Part(media=Media(url='data:image/png;base64,DATA'))) == {
            'type': 'image',
            'data': 'DATA',
            'mime_type': 'image/png',
        }

    def test_data_url_without_content_type_raises(self) -> None:
        with pytest.raises(GenkitError, match='Media part missing contentType') as raised:
            to_interaction_content(Part(media=Media(url='data:;base64,DATA')))
        assert raised.value.status == 'INVALID_ARGUMENT'

    def test_sends_metadata_processing_on_a_youtube_video(self) -> None:
        part = Part(media=Media(url='https://www.youtube.com/watch?v=abc123'), metadata={'processing': 'agentic'})
        assert to_interaction_content(part) == {
            'type': 'video',
            'uri': 'https://www.youtube.com/watch?v=abc123',
            'processing': 'agentic',
        }

    def test_sends_metadata_processing_on_inline_and_uri_videos(self) -> None:
        inline = Part(
            media=Media(url='data:video/mp4;base64,AAAA', content_type='video/mp4'),
            metadata={'processing': 'static'},
        )
        assert to_interaction_content(inline) == {
            'type': 'video',
            'mime_type': 'video/mp4',
            'data': 'AAAA',
            'processing': 'static',
        }
        files_api = 'https://generativelanguage.googleapis.com/v1beta/files/abc123'
        uri = Part(media=Media(url=files_api, content_type='video/mp4'), metadata={'processing': 'agentic'})
        assert to_interaction_content(uri) == {
            'type': 'video',
            'mime_type': 'video/mp4',
            'uri': files_api,
            'processing': 'agentic',
        }

    def test_ignores_metadata_processing_on_media_that_is_not_video(self) -> None:
        part = Part(
            media=Media(url='data:image/png;base64,AAAA', content_type='image/png'),
            metadata={'processing': 'agentic'},
        )
        assert to_interaction_content(part) == {'type': 'image', 'mime_type': 'image/png', 'data': 'AAAA'}


class TestToInteractionSteps:
    def test_tool_request(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(tool_request=ToolRequest(name='func', input={'a': 1}, ref='ref1'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'function_call', 'name': 'func', 'arguments': {'a': 1}, 'id': 'ref1'}
        ]

    def test_tool_response(self) -> None:
        messages = [
            Message(
                role='tool',
                content=[Part(tool_response=ToolResponse(name='func', output={'result': 'ok'}, ref='ref1'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'function_result', 'name': 'func', 'result': {'result': 'ok'}, 'call_id': 'ref1'}
        ]

    def test_model_output_grouping(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part.from_text('Thinking'), Part.from_text('Done')],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'model_output',
                'content': [{'type': 'text', 'text': 'Thinking'}, {'type': 'text', 'text': 'Done'}],
            }
        ]

    def test_system_role_rejected(self) -> None:
        messages = [Message(role='system', content=[Part.from_text('be terse')])]
        with pytest.raises(GenkitError, match='system_instruction') as raised:
            to_interaction_steps(messages)
        assert raised.value.status == 'INVALID_ARGUMENT'

    def test_code_execution_call_sends_the_language_lower_case(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part.from_custom(
                        {'executableCode': {'code': 'print(1)', 'language': 'PYTHON'}}, metadata={'callId': 'c1'}
                    )
                ],
            )
        ]
        # The Interactions API only accepts lower-case values.
        assert to_interaction_steps(messages) == [
            {
                'type': 'code_execution_call',
                'id': 'c1',
                'arguments': {'code': 'print(1)', 'language': 'python'},
            }
        ]

    def test_continues_code_execution_history_saved_from_generate_content(self) -> None:
        # The part shapes the generateContent path produces: no callId, and
        # language 'PYTHON', which the Interactions API rejects (400 "Supported
        # values: 'python', 'bash'", seen live). Real generateContent
        # signatures are accepted on the steps (seen live), so they're kept.
        messages = [
            Message(role='user', content=[Part.from_text('Run some code.')]),
            Message(
                role='model',
                content=[
                    Part.from_custom(
                        {'executableCode': {'code': 'print(5117)', 'language': 'PYTHON'}},
                        metadata={'thoughtSignature': 'sig1'},
                    ),
                    Part.from_custom({'codeExecutionResult': {'outcome': 'OUTCOME_OK', 'output': '5117\n'}}),
                    Part.from_custom({'executableCode': {'code': 'print(1)'}}),
                    Part.from_custom({'codeExecutionResult': {'outcome': 'OUTCOME_OK', 'output': '1\n'}}),
                    Part.from_text('The answer is 5117.'),
                ],
            ),
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'user_input', 'content': [{'type': 'text', 'text': 'Run some code.'}]},
            {
                'type': 'code_execution_call',
                'id': 'genkit-code-execution-id-0',
                'arguments': {'code': 'print(5117)', 'language': 'python'},
                'signature': 'sig1',
            },
            {'type': 'code_execution_result', 'call_id': 'genkit-code-execution-id-0', 'result': '5117\n'},
            {
                'type': 'code_execution_call',
                'id': 'genkit-code-execution-id-1',
                'arguments': {'code': 'print(1)', 'language': 'python'},
            },
            {'type': 'code_execution_result', 'call_id': 'genkit-code-execution-id-1', 'result': '1\n'},
            {'type': 'model_output', 'content': [{'type': 'text', 'text': 'The answer is 5117.'}]},
        ]

    def test_keeps_existing_code_execution_ids_and_bash_and_gives_a_lone_result_an_orphan_id(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part.from_custom(
                        {'executableCode': {'code': 'ls', 'language': 'bash'}}, metadata={'callId': 'call_1'}
                    ),
                    Part.from_custom({'codeExecutionResult': {'output': 'a.txt\n'}}, metadata={'callId': 'call_1'}),
                    Part.from_custom({'codeExecutionResult': {'output': 'stray\n'}}),
                ],
            )
        ]
        steps = to_interaction_steps(messages)
        assert steps == [
            {'type': 'code_execution_call', 'id': 'call_1', 'arguments': {'code': 'ls', 'language': 'bash'}},
            {'type': 'code_execution_result', 'call_id': 'call_1', 'result': 'a.txt\n'},
            {'type': 'code_execution_result', 'call_id': 'genkit-code-execution-orphan-id-0', 'result': 'stray\n'},
        ]

    def test_code_execution_ids_do_not_change_the_callers_messages(self) -> None:
        part = Part.from_custom({'executableCode': {'code': 'print(1)'}})
        messages = [Message(role='model', content=[part])]
        to_interaction_steps(messages)
        assert messages[0].content[0].metadata is None

    def test_google_search_call(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part.from_custom(
                        {'googleSearchCall': {'id': 'gs1', 'arguments': {'queries': ['genkit']}}},
                        metadata={'thoughtSignature': 'sig'},
                    )
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'google_search_call',
                'id': 'gs1',
                'arguments': {'queries': ['genkit']},
                'signature': 'sig',
            }
        ]

    def test_thought_becomes_its_own_step(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part.from_reasoning('plan the answer', metadata={'thoughtSignature': 'sig-t'})],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'thought',
                'summary': [{'type': 'text', 'text': 'plan the answer'}],
                'signature': 'sig-t',
            }
        ]

    def test_mixed_model_turn_preserves_content_order(self) -> None:
        """Inline content flushes before a standalone step so order matches the message."""
        messages = [
            Message(
                role='model',
                content=[
                    Part.from_reasoning('think', metadata={'thoughtSignature': 's'}),
                    Part.from_text('calling tool'),
                    Part(tool_request=ToolRequest(name='lookup', input={'q': 1}, ref='c1')),
                    Part.from_text('after'),
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'thought',
                'summary': [{'type': 'text', 'text': 'think'}],
                'signature': 's',
            },
            {
                'type': 'model_output',
                'content': [{'type': 'text', 'text': 'calling tool'}],
            },
            {'type': 'function_call', 'name': 'lookup', 'arguments': {'q': 1}, 'id': 'c1'},
            {
                'type': 'model_output',
                'content': [{'type': 'text', 'text': 'after'}],
            },
        ]

    def test_unknown_step_round_trips(self) -> None:
        blob = {'type': 'future_step', 'payload': {'x': 1}}
        messages = [
            Message(
                role='model',
                content=[Part.from_custom({'unknownStep': blob})],
            )
        ]
        assert to_interaction_steps(messages) == [blob]

    def test_unknown_step_inbound_then_outbound(self) -> None:
        class FutureStep(BaseModel):
            type: str = 'future_step'
            payload: dict

        parts = from_interaction_step(cast(Any, FutureStep(payload={'x': 1})))
        root = parts[0]
        assert root.custom is not None
        assert (root.custom or {})['unknownStep']['type'] == 'future_step'
        assert to_interaction_steps([Message(role='model', content=parts)]) == [
            {'type': 'future_step', 'payload': {'x': 1}}
        ]

    def test_signed_multimodal_thought_replays_original_step(self) -> None:
        thought_dump = {
            'type': 'thought',
            'signature': 'sig-img',
            'summary': [
                {'type': 'text', 'text': 'see this'},
                {'type': 'image', 'uri': 'gs://bucket/thumb.png', 'mime_type': 'image/png'},
            ],
        }
        messages = [
            Message(
                role='model',
                content=[Part.from_custom({'thought': thought_dump}, metadata={'thoughtSignature': 'sig-img'})],
            )
        ]
        assert to_interaction_steps(messages) == [thought_dump]

    def test_thought_with_image_replays_the_original_step(self) -> None:
        """A thought whose summary has an image comes back as that same step."""
        step = ThoughtStep.model_validate({
            'type': 'thought',
            'signature': 'sig-img',
            'summary': [
                {'type': 'text', 'text': 'see this'},
                {'type': 'image', 'uri': 'gs://bucket/thumb.png', 'mime_type': 'image/png'},
            ],
        })
        part = from_thought_step(step)
        assert to_thought_step(part) == step.model_dump(mode='python')

    def test_json_array_function_result_is_boxed(self) -> None:
        messages = [
            Message(
                role='tool',
                content=[Part(tool_response=ToolResponse(name='fn', output=[1, 2], ref='r1'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'function_result',
                'name': 'fn',
                'result': {'result': [1, 2]},
                'call_id': 'r1',
            }
        ]

    def test_content_block_list_function_result_is_not_boxed(self) -> None:
        blocks = [
            {'type': 'text', 'text': 'hello'},
            {'type': 'image', 'uri': 'gs://x', 'mime_type': 'image/png'},
        ]
        messages = [
            Message(
                role='tool',
                content=[Part(tool_response=ToolResponse(name='fn', output=blocks, ref='r1'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'function_result',
                'name': 'fn',
                'result': blocks,
                'call_id': 'r1',
            }
        ]

    def test_failed_tool_result_keeps_is_error(self) -> None:
        messages = [
            Message(
                role='tool',
                content=[
                    Part(
                        tool_response=ToolResponse(name='fn', output={'e': 'boom'}, ref='r1'),
                        metadata={'isError': True},
                    )
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'function_result',
                'name': 'fn',
                'result': {'e': 'boom'},
                'call_id': 'r1',
                'is_error': True,
            }
        ]

    def test_scalar_tool_request_input_raises(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(tool_request=ToolRequest(name='lookup', input='Paris', ref='c1'))],
            )
        ]
        with pytest.raises(GenkitError, match='JSON object') as exc_info:
            to_interaction_steps(messages)
        assert exc_info.value.status == 'INVALID_ARGUMENT'

    def test_pydantic_tool_request_input_dumps(self) -> None:
        class City(BaseModel):
            city: str = 'Paris'

        messages = [
            Message(
                role='model',
                content=[Part(tool_request=ToolRequest(name='lookup', input=City(), ref='c1'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'function_call', 'name': 'lookup', 'arguments': {'city': 'Paris'}, 'id': 'c1'}
        ]

    def test_pydantic_tool_response_output_dumps(self) -> None:
        class Weather(BaseModel):
            temp: int = 72

        messages = [
            Message(
                role='tool',
                content=[Part(tool_response=ToolResponse(name='weather', output=Weather(), ref='r1'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'function_result', 'name': 'weather', 'result': {'temp': 72}, 'call_id': 'r1'}
        ]

    def test_nested_pydantic_tool_output_dumps(self) -> None:
        class Weather(BaseModel):
            temp: int = 72

        messages = [
            Message(
                role='tool',
                content=[Part(tool_response=ToolResponse(name='weather', output={'report': Weather()}, ref='r1'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'function_result', 'name': 'weather', 'result': {'report': {'temp': 72}}, 'call_id': 'r1'}
        ]


class TestFromInteractionContent:
    def test_text(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({
                'type': 'text',
                'text': 'Hello world',
                'annotations': [{'start_index': 0, 'end_index': 5, 'source': 'source'}],
            })
        )
        assert part_dict(result) == {
            'text': 'Hello world',
            'metadata': {
                'annotations': [{'start_index': 0, 'end_index': 5, 'source': 'source', 'type': 'file_citation'}]
            },
        }

    def test_image_data(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({'type': 'image', 'data': 'BASE64DATA', 'mime_type': 'image/png'})
        )
        assert part_dict(result) == {'media': {'url': 'data:image/png;base64,BASE64DATA', 'contentType': 'image/png'}}

    def test_image_resolution(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({
                'type': 'image',
                'uri': 'gs://bucket/image.png',
                'mime_type': 'image/png',
                'resolution': 'high',
            })
        )
        assert part_dict(result) == {
            'media': {'url': 'gs://bucket/image.png', 'contentType': 'image/png'},
            'metadata': {'resolution': 'high'},
        }

    def test_labels_audio_data_without_a_mime_type_as_l16(self) -> None:
        # Streamed single-voice TTS deltas (no mime_type) assembled into the
        # final response's content.
        content_type = 'audio/l16; rate=24000; channels=1'
        result = from_interaction_content(
            ContentAdapter.validate_python({'type': 'audio', 'data': 'AAEC', 'channels': 1, 'sample_rate': 24000})
        )
        assert part_dict(result) == {
            'media': {'url': f'data:{content_type};base64,AAEC', 'contentType': content_type}
        }

    def test_audio_without_a_mime_type_uses_the_sample_rate_and_channels_sent(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({'type': 'audio', 'data': 'AAEC', 'channels': 2, 'sample_rate': 16000})
        )
        assert result.media is not None
        assert result.media.content_type == 'audio/l16; rate=16000; channels=2'

    def test_audio_without_a_mime_type_defaults_to_24khz_mono(self) -> None:
        result = from_interaction_content(ContentAdapter.validate_python({'type': 'audio', 'data': 'AAEC'}))
        assert result.media is not None
        assert result.media.content_type == 'audio/l16; rate=24000; channels=1'

    def test_audio_mime_type_sent_by_the_server_wins(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({
                'type': 'audio',
                'data': 'AAEC',
                'mime_type': 'audio/wav',
                'sample_rate': 16000,
            })
        )
        assert part_dict(result) == {'media': {'url': 'data:audio/wav;base64,AAEC', 'contentType': 'audio/wav'}}

    def test_audio_uri_without_a_mime_type_is_not_labelled(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({'type': 'audio', 'uri': 'gs://bucket/clip.mp3'})
        )
        assert part_dict(result) == {'media': {'url': 'gs://bucket/clip.mp3'}}

    def test_thought(self) -> None:
        step = ThoughtStep.model_validate({
            'type': 'thought',
            'signature': 'SIG',
            'summary': [{'type': 'text', 'text': 'Thinking...'}],
        })
        result = from_thought_step(step)
        dumped = part_dict(result)
        assert dumped['reasoning'] == 'Thinking...'
        assert dumped['metadata'] == {'thoughtSignature': 'SIG'}
        assert dumped['custom']['thought']['type'] == 'thought'
        assert dumped['custom']['thought']['summary'] == [{'type': 'text', 'text': 'Thinking...'}]


class TestFromInteractionStep:
    def test_model_output_includes_empty_annotations(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({'type': 'model_output', 'content': [{'type': 'text', 'text': 'Hello'}]})
        )
        root = result[0]
        assert root.text is not None
        assert root.text == 'Hello'
        assert root.metadata is not None
        assert 'annotations' in root.metadata

    def test_text_annotations_are_plain_data(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'model_output',
                'content': [
                    {
                        'type': 'text',
                        'text': 'hi',
                        'annotations': [{'type': 'url_citation', 'url': 'https://x.example'}],
                    }
                ],
            })
        )
        root = result[0]
        assert root.text is not None
        anns = (root.metadata or {}).get('annotations')
        assert isinstance(anns, list) and anns
        assert isinstance(anns[0], dict)
        assert anns[0].get('type') == 'url_citation'
        assert anns[0].get('url') == 'https://x.example'

    def test_user_input_dropped(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({'type': 'user_input', 'content': [{'type': 'text', 'text': 'Hello'}]})
        )
        assert result == []

    def test_function_call_is_tool_request_when_pending_and_custom_otherwise(self) -> None:
        step = StepAdapter.validate_python({
            'type': 'function_call',
            'id': 'c1',
            'name': 'get_weather',
            'arguments': {'city': 'Austin'},
        })
        pending = from_interaction_step(step, is_pending=True)[0]
        assert pending.tool_request is not None
        assert pending.tool_request.name == 'get_weather'
        assert pending.tool_request.input == {'city': 'Austin'}
        assert pending.tool_request.ref == 'c1'

        resolved = from_interaction_step(step)[0]
        assert resolved.tool_request is None
        assert resolved.custom == {
            'serverFunctionCall': {'id': 'c1', 'name': 'get_weather', 'arguments': {'city': 'Austin'}}
        }

    def test_function_result_is_custom_stash_not_tool_response(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'function_result',
                'call_id': 'c1',
                'name': 'get_weather',
                'result': {'temp': 92},
                'is_error': False,
            })
        )
        root = result[0]
        assert root.custom is not None
        assert root.tool_response is None
        stash = (root.custom or {})['serverFunctionResult']
        assert stash['name'] == 'get_weather'
        assert stash['result'] == {'temp': 92}
        assert stash['call_id'] == 'c1'
        assert stash['is_error'] is False

    def test_failed_function_result_keeps_is_error_on_stash(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'function_result',
                'call_id': 'c1',
                'name': 'get_weather',
                'result': {'e': 'boom'},
                'is_error': True,
            })
        )
        root = result[0]
        assert root.custom is not None
        stash = (root.custom or {})['serverFunctionResult']
        assert stash['is_error'] is True
        again = to_interaction_steps([Message(role='model', content=result)])
        assert again == [
            {
                'type': 'function_result',
                'name': 'get_weather',
                'result': {'e': 'boom'},
                'call_id': 'c1',
                'is_error': True,
            }
        ]

    def test_google_search_call_is_custom_part(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'google_search_call',
                'id': 'gs1',
                'arguments': {'queries': ['genkit']},
                'signature': 'sig',
            })
        )
        root = result[0]
        assert root.custom is not None
        assert root.custom == {'googleSearchCall': {'id': 'gs1', 'arguments': {'queries': ['genkit']}}}
        assert root.metadata == {'thoughtSignature': 'sig'}

    def test_code_execution_call_is_custom_part(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'code_execution_call',
                'id': 'ce1',
                'arguments': {'code': 'print(1)', 'language': 'python'},
            })
        )
        root = result[0]
        assert root.custom is not None
        assert root.custom == {'executableCode': {'code': 'print(1)', 'language': 'python'}}
        assert root.metadata == {'callId': 'ce1'}


class TestInboundFlattensToSingleModelMessage:
    """Every Interaction response becomes one role=model Message of flattened parts."""

    def test_mixed_tape_drops_user_input_and_keeps_order(self) -> None:
        interaction = Interaction.model_validate({
            'id': 'ix-1',
            'status': 'completed',
            'environment_id': 'env-1',
            'steps': [
                {'type': 'user_input', 'content': [{'type': 'text', 'text': 'weather?'}]},
                {
                    'type': 'thought',
                    'signature': 'sig',
                    'summary': [{'type': 'text', 'text': 'need a tool'}],
                },
                {'type': 'model_output', 'content': [{'type': 'text', 'text': 'checking'}]},
                {
                    'type': 'function_call',
                    'id': 'c1',
                    'name': 'get_weather',
                    'arguments': {'city': 'Austin'},
                },
            ],
        })
        op = from_interaction(interaction)
        assert op.done is True
        assert op.output is not None
        message = op.output.message
        assert message is not None
        assert message.role == 'model'
        assert message.metadata == {
            'interactionId': 'ix-1',
            'environmentId': 'env-1',
            'interactionStatus': 'completed',
        }
        assert message.content[0].reasoning is not None
        assert message.content[1].text is not None
        assert message.content[2].custom == {
            'serverFunctionCall': {'id': 'c1', 'name': 'get_weather', 'arguments': {'city': 'Austin'}}
        }
        assert message.content[0].reasoning == 'need a tool'
        assert message.content[1].text == 'checking'

    def test_already_resolved_tool_pair_stays_inside_model_message(self) -> None:
        """A tape that already has function_result stashes both call and result as custom, never tool_request."""
        steps = [
            StepAdapter.validate_python({
                'type': 'function_call',
                'id': 'c1',
                'name': 'write_to_file',
                'arguments': {'path': 'hello.py'},
            }),
            StepAdapter.validate_python({
                'type': 'function_result',
                'call_id': 'c1',
                'name': 'write_to_file',
                'result': {'ok': True},
            }),
            StepAdapter.validate_python({
                'type': 'model_output',
                'content': [{'type': 'text', 'text': 'Wrote hello.py'}],
            }),
        ]
        parts = parts_from_steps(steps, status='requires_action')
        assert parts[0].tool_request is None
        assert parts[0].custom == {
            'serverFunctionCall': {'id': 'c1', 'name': 'write_to_file', 'arguments': {'path': 'hello.py'}}
        }
        assert parts[1].custom is not None
        assert parts[2].text is not None
        stash = parts[1].custom['serverFunctionResult']
        assert stash['name'] == 'write_to_file'
        assert stash['result'] == {'ok': True}


class TestFunctionCallRoundTrip:
    """Pending function_call steps use ToolRequestPart; server-resolved ones round-trip via serverFunctionCall."""

    def test_outbound_then_inbound_preserves_tool_request(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(tool_request=ToolRequest(name='get_weather', input={'city': 'Austin'}, ref='c1'))],
            )
        ]
        steps = to_interaction_steps(messages)
        assert steps == [{'type': 'function_call', 'name': 'get_weather', 'arguments': {'city': 'Austin'}, 'id': 'c1'}]
        inbound = from_interaction_step(StepAdapter.validate_python(steps[0]), is_pending=True)
        root = inbound[0]
        assert root.tool_request is not None
        assert root.tool_request.name == 'get_weather'
        assert root.tool_request.input == {'city': 'Austin'}
        assert root.tool_request.ref == 'c1'

    def test_inbound_then_outbound_preserves_function_call(self) -> None:
        step = StepAdapter.validate_python({
            'type': 'function_call',
            'id': 'c1',
            'name': 'get_weather',
            'arguments': {'city': 'Austin'},
            'signature': 'sig-call',
        })
        pending_part = from_interaction_step(step, is_pending=True)[0]
        assert pending_part.tool_request is not None
        assert pending_part.metadata == {'thoughtSignature': 'sig-call'}
        assert to_interaction_steps([Message(role='model', content=[pending_part])]) == [
            {
                'type': 'function_call',
                'name': 'get_weather',
                'arguments': {'city': 'Austin'},
                'id': 'c1',
                'signature': 'sig-call',
            }
        ]

        server_part = from_interaction_step(step, is_pending=False)[0]
        assert server_part.tool_request is None
        assert server_part.custom == {
            'serverFunctionCall': {'id': 'c1', 'name': 'get_weather', 'arguments': {'city': 'Austin'}}
        }
        assert server_part.metadata == {'thoughtSignature': 'sig-call'}
        assert to_interaction_steps([Message(role='model', content=[server_part])]) == [
            {
                'type': 'function_call',
                'name': 'get_weather',
                'arguments': {'city': 'Austin'},
                'id': 'c1',
                'signature': 'sig-call',
            }
        ]

    def test_thought_round_trip_keeps_signature(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part.from_reasoning('plan', metadata={'thoughtSignature': 'sig'})],
            )
        ]
        steps = to_interaction_steps(messages)
        inbound = from_interaction_step(StepAdapter.validate_python(steps[0]))
        root = inbound[0]
        assert root.reasoning is not None
        assert root.reasoning == 'plan'
        assert root.metadata == {'thoughtSignature': 'sig'}


def test_operation_does_not_persist_client_options() -> None:
    op = from_interaction(Interaction.model_validate({'id': '123', 'status': 'in_progress'}))

    assert not op.metadata


class TestFromInteractionStatusMapping:
    def test_cancelled(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'cancelled'}))
        assert result.done is True
        assert result.output is not None
        assert result.output.finish_reason == 'aborted'
        assert result.output.finish_message == 'Operation cancelled'
        assert part_dict(result.output.message.content[0]) == {'text': 'Operation cancelled.'}

    def test_failed_exits_poll_loop(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'failed',
                'steps': [{'type': 'model_output', 'error': {'code': 3, 'message': 'boom'}}],
            })
        )
        assert result.done is True
        assert result.error is not None
        assert result.error.message == 'boom'

    def test_failed_without_step_error_falls_back(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'failed'}))
        assert result.done is True
        assert result.error is not None
        assert result.error.message == 'Interaction failed'

    def test_failed_uses_top_level_errors(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'failed',
                'errors': [{'code': '400', 'message': 'Invalid prompt format'}],
            })
        )
        assert result.done is True
        assert result.error is not None
        assert result.error.message == 'Invalid prompt format'

    def test_failed_content_block_is_a_blocked_output(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'failed',
                'errors': [{'code': 'recitation', 'message': 'Recitation detected.'}],
            })
        )
        assert result.done is True
        assert result.error is None
        assert result.output is not None
        assert result.output.finish_reason == 'blocked'
        assert result.output.finish_message == 'Recitation detected.'

    def test_requires_action_exits_poll_loop(self) -> None:
        interaction = Interaction.model_validate({
            'id': '123',
            'status': 'requires_action',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'approve plan'}]}],
        })
        result = from_interaction(interaction)
        assert result.id == '123'
        assert result.done is True
        assert result.output is None
        assert result.error is not None
        assert result.error.message == 'Background models do not support interrupts'

    def test_in_progress(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'in_progress'}))
        assert result.done is False

    def test_queued_keeps_polling(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'queued'}))
        assert result.done is False

    def test_completed_empty_steps_still_has_output(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'completed', 'steps': []}))
        assert result.done is True
        assert result.error is None
        assert result.output is not None
        assert result.output.finish_reason == 'stop'
        assert result.output.message is not None
        assert result.output.message.content == []

    def test_unknown_status_exits_poll_loop(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'expired'}))
        assert result.done is True
        assert result.output is None
        assert result.error is not None
        assert 'expired' in (result.error.message or '')

    def test_incomplete_surfaces_partial_steps(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'incomplete',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'partial'}]}],
            })
        )
        assert result.done is True
        assert result.error is None
        assert result.output is not None
        assert result.output.finish_reason == 'length'
        assert result.output.finish_message == 'Interaction incomplete (truncated output)'
        assert result.output.message is not None
        assert result.output.message.content[0].text == 'partial'
        assert result.output.message.metadata is not None
        assert result.output.message.metadata.get('interactionStatus') == 'incomplete'

    def test_incomplete_without_steps_errors(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'incomplete'}))
        assert result.done is True
        assert result.output is None
        assert result.error is not None

    def test_budget_exceeded_surfaces_partial_steps(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'budget_exceeded',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'draft'}]}],
            })
        )
        assert result.done is True
        assert result.error is None
        assert result.output is not None
        assert result.output.finish_reason == 'aborted'
        assert result.output.finish_message == 'Interaction exceeded its budget'
        assert result.output.message is not None
        assert result.output.message.content[0].text == 'draft'
        assert result.output.message.metadata is not None
        assert result.output.message.metadata.get('interactionStatus') == 'budget_exceeded'

    def test_budget_exceeded_without_steps_errors(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'budget_exceeded'}))
        assert result.done is True
        assert result.output is None
        assert result.error is not None
        assert 'budget' in (result.error.message or '').lower()


class TestFromInteractionSync:
    def test_failed_raises(self) -> None:
        with pytest.raises(GenkitError, match='Interaction failed') as raised:
            from_interaction_sync(Interaction.model_validate({'status': 'failed'}))
        assert raised.value.status == 'UNKNOWN'
        assert str(raised.value) == 'UNKNOWN: Interaction failed'

    @pytest.mark.parametrize(
        ('code', 'status'),
        [
            ('RESOURCE_EXHAUSTED', 'RESOURCE_EXHAUSTED'),
            ('resource_exhausted', 'RESOURCE_EXHAUSTED'),
            ('429', 'RESOURCE_EXHAUSTED'),
            ('503', 'UNAVAILABLE'),
            ('3', 'INVALID_ARGUMENT'),
        ],
    )
    def test_failed_with_provider_code_is_classified(self, code: str, status: str) -> None:
        """A quota failure the server reported keeps its status so retry and fallback can act on it."""
        interaction = Interaction.model_validate({
            'id': 'ix-1',
            'status': 'failed',
            'errors': [{'code': code, 'message': 'Quota exceeded for lyria'}],
        })
        with pytest.raises(GenkitError, match='Quota exceeded for lyria') as raised:
            from_interaction_sync(interaction)
        assert raised.value.status == status
        assert isinstance(raised.value.__cause__, ValueError)

    def test_failed_step_error_grpc_code_is_classified(self) -> None:
        interaction = Interaction.model_validate({
            'id': 'ix-1',
            'status': 'failed',
            'steps': [{'type': 'model_output', 'error': {'code': 8, 'message': 'out of quota'}}],
        })
        with pytest.raises(GenkitError, match='out of quota') as raised:
            from_interaction_sync(interaction)
        assert raised.value.status == 'RESOURCE_EXHAUSTED'

    @pytest.mark.parametrize('code', ['UNKNOWN', 'OK', '0', '2', '200', '999', 'SOMETHING_NEW', ''])
    def test_failed_with_unusable_code_is_unknown(self, code: str) -> None:
        """No real status from the server: UNKNOWN, which default Retry and Fallback leave alone."""
        interaction = Interaction.model_validate({
            'id': 'ix-1',
            'status': 'failed',
            'errors': [{'code': code, 'message': 'model crashed'}],
        })
        with pytest.raises(GenkitError) as raised:
            from_interaction_sync(interaction)
        assert raised.value.status == 'UNKNOWN'
        assert str(raised.value) == 'UNKNOWN: model crashed'

    def test_returns_finish_reason_blocked_for_a_post_execution_safety_block(self) -> None:
        result = from_interaction_sync(
            Interaction.model_validate({
                'id': 'int-1',
                'status': 'failed',
                'errors': [{'code': 'safety', 'message': 'Response blocked due to safety violations.'}],
            })
        )
        assert result.finish_reason == 'blocked'
        assert result.finish_message == 'Response blocked due to safety violations.'
        assert result.message is not None
        assert result.message.content == []
        assert result.message.metadata is not None
        assert result.message.metadata.get('interactionId') == 'int-1'

    @pytest.mark.parametrize(
        'code',
        [
            'safety',
            'recitation',
            'language',
            'prohibited_content',
            'spii',
            'blocklist',
            'image_safety',
            'image_prohibited_content',
            'image_recitation',
            'image_other',
            'content_blocked',
            'jailbreak',
            'model_armor',
        ],
    )
    def test_returns_finish_reason_blocked_for_every_content_block_code(self, code: str) -> None:
        result = from_interaction_sync(
            Interaction.model_validate({'status': 'failed', 'errors': [{'code': code, 'message': 'blocked'}]})
        )
        assert result.finish_reason == 'blocked'

    def test_carries_usage_on_a_blocked_response(self) -> None:
        result = from_interaction_sync(
            Interaction.model_validate({
                'status': 'failed',
                'errors': [{'code': 'safety', 'message': 'blocked'}],
                'usage': {
                    'total_input_tokens': 10,
                    'total_output_tokens': 0,
                    'total_tokens': 10,
                    'input_tokens_by_modality': [
                        {'modality': 'text', 'tokens': 8},
                        {'modality': 'image', 'tokens': 2},
                    ],
                },
            })
        )
        assert result.finish_reason == 'blocked'
        assert result.usage is not None
        assert result.usage.model_dump(exclude_none=True) == {
            'inputTokens': 10,
            'outputTokens': 0,
            'totalTokens': 10,
            'custom': {'inputTextTokens': 8, 'inputImageTokens': 2},
        }

    def test_raises_a_retryable_aborted_error_for_malformed_function_call(self) -> None:
        advice = 'Model generated invalid JSON syntax and the output could not be parsed. Please retry the request.'
        with pytest.raises(GenkitError) as raised:
            from_interaction_sync(
                Interaction.model_validate({
                    'status': 'failed',
                    'errors': [{'code': 'malformed_function_call', 'message': advice}],
                })
            )
        assert raised.value.status == 'ABORTED'
        assert advice in str(raised.value)

    @pytest.mark.parametrize(
        ('code', 'status'),
        [
            ('unexpected_tool_call', 'ABORTED'),
            ('no_image', 'ABORTED'),
            ('service_unavailable', 'UNAVAILABLE'),
            ('quota_exceeded', 'RESOURCE_EXHAUSTED'),
            ('model_not_found', 'NOT_FOUND'),
            ('agent_max_token_limit', 'INVALID_ARGUMENT'),
        ],
    )
    def test_raises_the_status_for_an_interactions_error_code(self, code: str, status: str) -> None:
        with pytest.raises(GenkitError) as raised:
            from_interaction_sync(
                Interaction.model_validate({'status': 'failed', 'errors': [{'code': code, 'message': 'x'}]})
            )
        assert raised.value.status == status

    def test_queued_is_failed_precondition(self) -> None:
        with pytest.raises(GenkitError, match='still running') as raised:
            from_interaction_sync(Interaction.model_validate({'id': '123', 'status': 'queued'}))
        assert raised.value.status == 'FAILED_PRECONDITION'

    def test_requires_action_returns_pending_tool_requests_and_skips_resolved_sandbox_calls(self) -> None:
        interaction = Interaction.model_validate({
            'id': 'v1_abc123',
            'status': 'requires_action',
            'environment_id': 'env-123',
            'steps': [
                {
                    'type': 'function_call',
                    'id': 'sandbox-1',
                    'name': 'write_to_file',
                    'arguments': {'path': 'a.txt'},
                },
                {
                    'type': 'function_result',
                    'call_id': 'sandbox-1',
                    'name': 'write_to_file',
                    'result': {'ok': True},
                },
                {
                    'type': 'function_call',
                    'id': 'call-1',
                    'name': 'lookup_allergens',
                    'arguments': {'dish': 'ramen'},
                },
            ],
        })
        result = from_interaction_sync(interaction)
        assert result.finish_reason == 'stop'
        assert result.message is not None
        assert result.message.metadata == {
            'interactionId': 'v1_abc123',
            'environmentId': 'env-123',
            'interactionStatus': 'requires_action',
        }
        assert result.message.content[0].tool_request is None
        assert result.message.content[0].custom == {
            'serverFunctionCall': {'id': 'sandbox-1', 'name': 'write_to_file', 'arguments': {'path': 'a.txt'}}
        }
        assert result.message.content[1].custom is not None
        assert result.message.content[2].tool_request is not None
        assert result.message.content[2].tool_request.name == 'lookup_allergens'
        assert result.message.content[2].tool_request.ref == 'call-1'
        assert result.message.content[2].tool_request.input == {'dish': 'ramen'}

    def test_cancelled(self) -> None:
        result = from_interaction_sync(Interaction.model_validate({'id': '123', 'status': 'cancelled'}))
        assert result.finish_reason == 'aborted'
        assert result.finish_message == 'Operation cancelled'

    def test_incomplete_surfaces_partial_steps(self) -> None:
        result = from_interaction_sync(
            Interaction.model_validate({
                'id': '123',
                'status': 'incomplete',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'partial'}]}],
            })
        )
        assert result.finish_reason == 'length'
        assert result.finish_message == 'Interaction incomplete (truncated output)'
        assert result.message is not None
        assert result.message.content[0].text == 'partial'

    def test_incomplete_without_steps_raises(self) -> None:
        with pytest.raises(ValueError, match='incomplete'):
            from_interaction_sync(Interaction.model_validate({'id': '123', 'status': 'incomplete'}))

    def test_budget_exceeded_surfaces_partial_steps(self) -> None:
        result = from_interaction_sync(
            Interaction.model_validate({
                'id': '123',
                'status': 'budget_exceeded',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'draft'}]}],
            })
        )
        assert result.finish_reason == 'aborted'
        assert result.finish_message == 'Interaction exceeded its budget'

    def test_in_progress_raises(self) -> None:
        with pytest.raises(GenkitError, match='still running') as raised:
            from_interaction_sync(Interaction.model_validate({'id': '123', 'status': 'in_progress'}))
        assert raised.value.status == 'FAILED_PRECONDITION'

    def test_unknown_status_raises(self) -> None:
        with pytest.raises(ValueError, match='Unknown interaction status'):
            from_interaction_sync(Interaction.model_validate({'id': '123', 'status': 'expired'}))


class TestSplitSystemInstruction:
    def test_lifts_system_turns(self) -> None:
        instruction, turns = split_system_instruction([
            Message(role='system', content=[Part.from_text('be terse')]),
            Message(role='user', content=[Part.from_text('hi')]),
        ])
        assert instruction == 'be terse'
        assert len(turns) == 1
        assert turns[0].role == 'user'

    def test_joins_multiple_system_turns(self) -> None:
        instruction, turns = split_system_instruction([
            Message(role='system', content=[Part.from_text('one')]),
            Message(role='system', content=[Part.from_text('two')]),
        ])
        assert instruction == 'one\n\ntwo'
        assert turns == []


class TestUsageFromInteraction:
    def test_maps_totals_and_puts_modality_tokens_in_custom(self) -> None:
        usage = Usage.model_validate({
            'total_input_tokens': 10,
            'total_output_tokens': 4,
            'total_tokens': 14,
            'total_cached_tokens': 2,
            'total_thought_tokens': 1,
            'input_tokens_by_modality': [{'modality': 'video', 'tokens': 3}],
            'output_tokens_by_modality': [{'modality': 'video', 'tokens': 1}],
        })
        mapped = usage_from_interaction(usage)
        assert mapped.input_tokens == 10
        assert mapped.output_tokens == 4
        assert mapped.total_tokens == 14
        assert mapped.cached_content_tokens == 2
        assert mapped.thoughts_tokens == 1
        # Per-modality counts are tokens, not Genkit's media-part counts.
        assert mapped.custom == {'inputVideoTokens': 3, 'outputVideoTokens': 1}
        assert mapped.input_videos is None
        assert mapped.output_videos is None

    def test_text_and_image_tokens_are_not_characters_or_images(self) -> None:
        mapped = usage_from_interaction(
            Usage.model_validate({
                'total_input_tokens': 10,
                'total_output_tokens': 0,
                'total_tokens': 10,
                'input_tokens_by_modality': [
                    {'modality': 'text', 'tokens': 8},
                    {'modality': 'image', 'tokens': 2},
                ],
            })
        )
        assert mapped.custom == {'inputTextTokens': 8, 'inputImageTokens': 2}
        assert mapped.input_characters is None
        assert mapped.input_images is None

    def test_reports_tool_use_tokens_in_custom(self) -> None:
        mapped = usage_from_interaction(
            Usage.model_validate({
                'total_input_tokens': 416,
                'total_output_tokens': 393,
                'total_tool_use_tokens': 2054,
                'total_thought_tokens': 2117,
                'total_tokens': 4980,
            })
        )
        assert mapped.custom == {'toolUseTokens': 2054}

    def test_reports_grounding_tool_counts_in_custom(self) -> None:
        # `grounding_tool_count` as seen live (googleAI, Google Search).
        mapped = usage_from_interaction(
            Usage.model_validate({
                'total_tokens': 1011,
                'total_tool_use_tokens': 0,
                'grounding_tool_count': [
                    {'type': 'google_search', 'count': 2},
                    {'type': 'google_maps', 'count': 1},
                ],
            })
        )
        assert mapped.custom == {'googleSearchCount': 2, 'googleMapsCount': 1}

    def test_leaves_custom_unset_without_breakdowns(self) -> None:
        mapped = usage_from_interaction(Usage.model_validate({'total_input_tokens': 4, 'total_tokens': 4}))
        assert mapped.custom is None


class TestWithBasicUsageStats:
    request = [
        Message(
            role='user',
            content=[
                Part.from_text('Describe this'),
                Part(media=Media(url='data:image/png;base64,AAAA', content_type='image/png')),
            ],
        )
    ]

    def test_adds_character_and_media_counts_next_to_the_token_counts(self) -> None:
        response = from_interaction_sync(
            Interaction.model_validate({
                'id': 'int-1',
                'status': 'completed',
                'steps': [
                    {
                        'type': 'model_output',
                        'content': [
                            {'type': 'text', 'text': 'A cat.'},
                            {'type': 'audio', 'mime_type': 'audio/wav', 'data': 'AAAA'},
                        ],
                    }
                ],
                'usage': {
                    'total_input_tokens': 270,
                    'total_output_tokens': 30,
                    'total_tokens': 300,
                    'input_tokens_by_modality': [
                        {'modality': 'text', 'tokens': 12},
                        {'modality': 'image', 'tokens': 258},
                    ],
                },
            })
        )
        result = with_basic_usage_stats(response, self.request)
        assert result.usage is not None
        # Same keys and values as the JS test (Genkit models dump camelCase).
        assert result.usage.model_dump(exclude_none=True) == {
            'inputCharacters': len('Describe this'),
            'inputImages': 1,
            'inputVideos': 0,
            'inputAudioFiles': 0,
            'outputCharacters': len('A cat.'),
            'outputImages': 0,
            'outputVideos': 0,
            'outputAudioFiles': 1,
            'inputTokens': 270,
            'outputTokens': 30,
            'totalTokens': 300,
            'custom': {'inputTextTokens': 12, 'inputImageTokens': 258},
        }

    def test_counts_the_request_even_when_the_api_sent_no_usage(self) -> None:
        response = from_interaction_sync(Interaction.model_validate({'id': 'int-1', 'status': 'completed'}))
        result = with_basic_usage_stats(response, self.request)
        assert result.usage is not None
        assert result.usage.input_characters == len('Describe this')
        assert result.usage.input_images == 1
        assert result.usage.output_characters == 0
        assert result.usage.input_tokens is None
