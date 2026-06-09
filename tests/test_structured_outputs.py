from __future__ import annotations

import pytest
from pydantic import ValidationError

from am_mvt.extraction.schemas import (
    ExtractionResponse,
    LLM_EXTRACTION_SCHEMA,
    parse_extraction_response,
)
from am_mvt.ingestion.llm_source_screening import (
    SOURCE_SCREENING_SCHEMA,
    SourceScreeningResponse,
)


def test_extraction_schema_is_strict_and_requires_records() -> None:
    assert LLM_EXTRACTION_SCHEMA["additionalProperties"] is False
    assert LLM_EXTRACTION_SCHEMA["required"] == ["records"]
    record_schema = LLM_EXTRACTION_SCHEMA["$defs"]["ExtractionRecord"]
    assert record_schema["additionalProperties"] is False
    assert set(record_schema["required"]) == set(record_schema["properties"])


def test_extraction_response_is_validated_before_postprocessing() -> None:
    assert parse_extraction_response({"records": []}) == {"records": []}
    with pytest.raises(ValidationError):
        ExtractionResponse.model_validate({"records": [{}]})


def test_source_screening_schema_is_strict() -> None:
    assert SOURCE_SCREENING_SCHEMA["additionalProperties"] is False
    candidate_schema = SOURCE_SCREENING_SCHEMA["$defs"]["SourceCandidate"]
    assert candidate_schema["additionalProperties"] is False
    assert set(candidate_schema["required"]) == set(candidate_schema["properties"])

    with pytest.raises(ValidationError):
        SourceScreeningResponse.model_validate(
            {"candidates": [{"title": "Incomplete candidate"}]}
        )
