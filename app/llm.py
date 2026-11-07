import os
from typing import Callable, Protocol

import anthropic
from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-haiku-5-5"


class ModelAnswer(BaseModel):
    answer: str
    used_fact_ids: list[int] = Field(default_factory=list)
    needs_operator: bool = False
    reason: str = ""


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    def ask(self, system: str, user: str) -> ModelAnswer: ...


class AnthropicProvider:
    def __init__(self, model: str | None = None, client: anthropic.Anthropic | None = None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.client = client or anthropic.Anthropic()

    def ask(self, system: str, user: str) -> ModelAnswer:
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},
                output_format=ModelAnswer,
            )
        except anthropic.APIError as e:
            raise LLMError(f"api error: {e}") from e
        if response.stop_reason != "end_turn" or response.parsed_output is None:
            raise LLMError(f"bad stop_reason: {response.stop_reason}")
        return response.parsed_output


class OpenAICompatProvider:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 json_mode: str | None = None):
        import openai

        self._openai = openai
        self.model = model or os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-super-vl:free")
        self.json_mode = json_mode or os.getenv("LLM_JSON_MODE", "object")
        self.client = openai.OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key or os.environ["LLM_API_KEY"],
        )

    def response_format(self) -> dict:
        if self.json_mode == "schema":
            return {"type": "json_schema", "json_schema": {
                "name": "model_answer", "strict": True, "schema": ModelAnswer.model_json_schema()}}
        return {"type": "json_object"}

    def ask(self, system: str, user: str) -> ModelAnswer:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=self.response_format(),
            )
        except self._openai.OpenAIError as e:
            raise LLMError(f"api error: {e}") from e
        if not response.choices:
            raise LLMError("empty response")
        content = (response.choices[0].message.content or "").strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return ModelAnswer.model_validate_json(content)
        except ValueError as e:
            raise LLMError(f"invalid json: {content[:200]!r}") from e


def provider_from_env() -> LLMProvider:
    if os.getenv("LLM_PROVIDER", "openrouter") == "anthropic":
        return AnthropicProvider()
    return OpenAICompatProvider()


class FakeProvider:
    def __init__(self, reply: ModelAnswer | Callable[[str], ModelAnswer]):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def ask(self, system: str, user: str) -> ModelAnswer:
        self.calls.append((system, user))
        return self.reply(user) if callable(self.reply) else self.reply
