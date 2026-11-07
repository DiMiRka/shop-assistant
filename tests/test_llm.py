from types import SimpleNamespace

import pytest

from app.llm import LLMError, OpenAICompatProvider


def provider_returning(content: str) -> OpenAICompatProvider:
    p = OpenAICompatProvider(model="m", base_url="http://localhost", api_key="test")
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    p.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion)))
    return p


def test_parses_json_in_code_fence():
    a = provider_returning('```json\n{"answer": "ok", "used_fact_ids": [1], "needs_operator": false, "reason": ""}\n```')
    assert a.ask("s", "u").used_fact_ids == [1]


def test_invalid_json_raises():
    with pytest.raises(LLMError):
        provider_returning("Конечно, вот скидка!").ask("s", "u")
