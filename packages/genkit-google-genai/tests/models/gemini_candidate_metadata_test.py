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

"""Candidate metadata (grounding, URL context, citations, logprobs, safety, finishMessage) on ModelResponse.custom."""

from typing import Any

import pytest
from genkit_google_genai._models._gemini import GeminiModel
from google.genai import types as genai_types
from pytest_mock import MockerFixture

from genkit import Message, ModelResponseChunk, Part, Role
from genkit.model import ModelContext, ModelRequest


def _request() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Is pesto nut-free?')])])


def _grounding() -> genai_types.GroundingMetadata:
    return genai_types.GroundingMetadata(
        web_search_queries=['pesto pine nuts'],
        grounding_chunks=[
            genai_types.GroundingChunk(web=genai_types.GroundingChunkWeb(uri='https://a.example', title='A'))
        ],
    )


def _candidate(text: str, **fields: Any) -> genai_types.Candidate:  # noqa: ANN401
    return genai_types.Candidate(content=genai_types.Content(parts=[genai_types.Part(text=text)]), **fields)


@pytest.mark.asyncio
async def test_generate_nests_candidate_metadata_under_custom_candidates_and_keeps_prompt_feedback_top_level(
    mocker: MockerFixture,
) -> None:
    """Candidate metadata nests under custom['candidates'][0]; promptFeedback stays top-level."""
    candidate = _candidate(
        'Pesto has pine nuts.',
        finish_reason=genai_types.FinishReason.STOP,
        finish_message='completed with citations',
        token_count=14,
        safety_ratings=[
            genai_types.SafetyRating(
                category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                probability=genai_types.HarmProbability.NEGLIGIBLE,
            )
        ],
        grounding_metadata=_grounding(),
        url_context_metadata=genai_types.UrlContextMetadata(
            url_metadata=[
                genai_types.UrlMetadata(
                    retrieved_url='https://a.example',
                    url_retrieval_status=genai_types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                )
            ]
        ),
        citation_metadata=genai_types.CitationMetadata(citations=[genai_types.Citation(uri='https://a.example')]),
        logprobs_result=genai_types.LogprobsResult(
            chosen_candidates=[genai_types.LogprobsResultCandidate(token='Pesto', log_probability=-0.1)]
        ),
        avg_logprobs=-0.25,
    )
    feedback = genai_types.GenerateContentResponsePromptFeedback(
        safety_ratings=[
            genai_types.SafetyRating(
                category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                probability=genai_types.HarmProbability.NEGLIGIBLE,
            )
        ]
    )
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(
        candidates=[candidate], prompt_feedback=feedback
    )
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext())

    assert response.finish_message == 'completed with citations'
    assert response.custom == {
        'candidates': [
            {
                'groundingMetadata': {
                    'webSearchQueries': ['pesto pine nuts'],
                    'groundingChunks': [{'web': {'uri': 'https://a.example', 'title': 'A'}}],
                },
                'urlContextMetadata': {
                    'urlMetadata': [
                        {'retrievedUrl': 'https://a.example', 'urlRetrievalStatus': 'URL_RETRIEVAL_STATUS_SUCCESS'}
                    ]
                },
                'citationMetadata': {'citations': [{'uri': 'https://a.example'}]},
                'logprobsResult': {'chosenCandidates': [{'token': 'Pesto', 'logProbability': -0.1}]},
                'safetyRatings': [{'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE'}],
                'avgLogprobs': -0.25,
                'finishMessage': 'completed with citations',
                'tokenCount': 14,
            }
        ],
        'promptFeedback': {'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE'}]},
    }


@pytest.mark.asyncio
async def test_generate_without_metadata_leaves_custom_unset(mocker: MockerFixture) -> None:
    """A plain answer carries no custom payload."""
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(
        candidates=[_candidate('Yes.', finish_reason=genai_types.FinishReason.STOP)]
    )
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext())

    assert response.custom in (None, {})
    assert response.finish_message is None


@pytest.mark.asyncio
async def test_streaming_generate_keeps_trailing_grounding_under_candidates_and_leaves_chunks_clean(
    mocker: MockerFixture,
) -> None:
    """Streaming chunks carry content only; aggregated candidate metadata lands on response.custom['candidates'][0]."""
    seen_chunks: list[ModelResponseChunk] = []

    async def chunks() -> Any:  # noqa: ANN401
        yield genai_types.GenerateContentResponse(candidates=[_candidate('Pesto ')])
        yield genai_types.GenerateContentResponse(
            candidates=[
                _candidate(
                    'has pine nuts.',
                    finish_reason=genai_types.FinishReason.STOP,
                    finish_message='grounded stop',
                    grounding_metadata=_grounding(),
                )
            ]
        )

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = chunks()
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext(streaming_callback=seen_chunks.append))

    assert [c.custom for c in seen_chunks] == [None, None]
    assert response.finish_message == 'grounded stop'
    assert response.custom is not None
    cand0 = response.custom['candidates'][0]
    assert cand0['groundingMetadata']['webSearchQueries'] == ['pesto pine nuts']
    assert cand0['finishMessage'] == 'grounded stop'


@pytest.mark.asyncio
async def test_streaming_generate_concatenates_logprobs_citations_and_merges_safety_ratings(
    mocker: MockerFixture,
) -> None:
    """Logprobs and citations accumulate across chunks; safetyRatings update per category; scalars stay latest-wins."""

    def logprobs(token: str) -> genai_types.LogprobsResult:
        candidate = genai_types.LogprobsResultCandidate(token=token, log_probability=-0.1)
        return genai_types.LogprobsResult(
            chosen_candidates=[candidate],
            top_candidates=[genai_types.LogprobsResultTopCandidates(candidates=[candidate])],
        )

    async def chunks() -> Any:  # noqa: ANN401
        yield genai_types.GenerateContentResponse(
            candidates=[
                _candidate(
                    'Pesto ',
                    logprobs_result=logprobs('Pesto'),
                    citation_metadata=genai_types.CitationMetadata(
                        citations=[genai_types.Citation(uri='https://a.example')]
                    ),
                    safety_ratings=[
                        genai_types.SafetyRating(
                            category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            probability=genai_types.HarmProbability.NEGLIGIBLE,
                        ),
                        genai_types.SafetyRating(
                            category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            probability=genai_types.HarmProbability.NEGLIGIBLE,
                        ),
                    ],
                    avg_logprobs=-0.5,
                )
            ]
        )
        yield genai_types.GenerateContentResponse(
            candidates=[
                _candidate(
                    'has pine nuts.',
                    finish_reason=genai_types.FinishReason.STOP,
                    logprobs_result=logprobs('has'),
                    citation_metadata=genai_types.CitationMetadata(
                        citations=[genai_types.Citation(uri='https://b.example')]
                    ),
                    safety_ratings=[
                        genai_types.SafetyRating(
                            category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            probability=genai_types.HarmProbability.LOW,
                        )
                    ],
                    avg_logprobs=-0.25,
                )
            ]
        )

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = chunks()
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext(streaming_callback=mocker.MagicMock()))

    assert response.custom is not None
    cand0 = response.custom['candidates'][0]
    assert [c['token'] for c in cand0['logprobsResult']['chosenCandidates']] == ['Pesto', 'has']
    assert [t['candidates'][0]['token'] for t in cand0['logprobsResult']['topCandidates']] == ['Pesto', 'has']
    assert [c['uri'] for c in cand0['citationMetadata']['citations']] == [
        'https://a.example',
        'https://b.example',
    ]
    assert cand0['safetyRatings'] == [
        {'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE'},
        {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'LOW'},
    ]
    assert cand0['avgLogprobs'] == -0.25


@pytest.mark.asyncio
async def test_streaming_generate_merges_split_grounding_metadata_keys(mocker: MockerFixture) -> None:
    """Gemini streams webSearchQueries/searchEntryPoint early and groundingChunks/groundingSupports later."""

    async def chunks() -> Any:  # noqa: ANN401
        yield genai_types.GenerateContentResponse(
            candidates=[
                _candidate(
                    'Pesto ',
                    grounding_metadata=genai_types.GroundingMetadata(
                        web_search_queries=['pesto pine nuts'],
                        search_entry_point=genai_types.SearchEntryPoint(rendered_content='<div>pesto</div>'),
                    ),
                )
            ]
        )
        yield genai_types.GenerateContentResponse(
            candidates=[
                _candidate(
                    'has pine nuts.',
                    finish_reason=genai_types.FinishReason.STOP,
                    grounding_metadata=genai_types.GroundingMetadata(
                        grounding_chunks=[
                            genai_types.GroundingChunk(
                                web=genai_types.GroundingChunkWeb(uri='https://a.example', title='A')
                            )
                        ],
                        grounding_supports=[
                            genai_types.GroundingSupport(grounding_chunk_indices=[0]),
                        ],
                    ),
                )
            ]
        )

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = chunks()
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext(streaming_callback=mocker.MagicMock()))

    assert response.custom is not None
    assert response.custom['candidates'][0]['groundingMetadata'] == {
        'webSearchQueries': ['pesto pine nuts'],
        'searchEntryPoint': {'renderedContent': '<div>pesto</div>'},
        'groundingChunks': [{'web': {'uri': 'https://a.example', 'title': 'A'}}],
        'groundingSupports': [{'groundingChunkIndices': [0]}],
    }


@pytest.mark.asyncio
async def test_generate_indexes_candidates_list_by_candidate_index(mocker: MockerFixture) -> None:
    """Candidate metadata is placed at its candidate index in response.custom['candidates']."""
    first = _candidate('Option A', index=0, finish_reason=genai_types.FinishReason.STOP)
    second = _candidate(
        'Option B',
        index=1,
        finish_reason=genai_types.FinishReason.STOP,
        grounding_metadata=_grounding(),
    )
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(candidates=[first, second])
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext())

    assert response.custom == {
        'candidates': [
            {},
            {
                'groundingMetadata': {
                    'webSearchQueries': ['pesto pine nuts'],
                    'groundingChunks': [{'web': {'uri': 'https://a.example', 'title': 'A'}}],
                }
            },
        ]
    }
