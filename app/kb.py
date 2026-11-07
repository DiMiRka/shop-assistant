from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

KB_PATH = Path(__file__).with_name("kb.yaml")


class Fact(BaseModel):
    id: int
    topic: str
    fact: str


class KnowledgeBase(BaseModel):
    facts: list[Fact]

    def get(self, fact_id: int) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)

    def as_prompt(self) -> str:
        return "\n".join(f"[{f.id}] {f.topic}: {f.fact}" for f in self.facts)


@lru_cache
def load_kb(path: Path = KB_PATH) -> KnowledgeBase:
    return KnowledgeBase.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
