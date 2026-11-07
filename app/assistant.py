import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.guard import check, order_numbers
from app.kb import KnowledgeBase, load_kb
from app.llm import LLMError, LLMProvider, ModelAnswer
from app.prompt import build_system_prompt, build_user_message

log = logging.getLogger(__name__)

FALLBACK = "Точной информации по этому вопросу у меня нет, а гадать не буду. Передам ваш вопрос менеджеру, он ответит."
FALLBACK_WITH_ORDER = "Статус заказа {order} я не вижу. Передам номер {order} менеджеру, он уточнит и ответит вам."


@dataclass
class Reply:
    text: str
    used_fact_ids: list[int]
    needs_operator: bool
    blocked: bool = False
    violations: list[str] = field(default_factory=list)
    llm_failed: bool = False
    model_answer: ModelAnswer | None = None


class Assistant:
    def __init__(self, llm: LLMProvider, kb: KnowledgeBase | None = None):
        self.llm = llm
        self.kb = kb or load_kb()
        self.system = build_system_prompt(self.kb)

    def fallback(self, client_text: str, violations: list[str], model_answer: ModelAnswer | None) -> Reply:
        orders = sorted(order_numbers(client_text))
        text = FALLBACK_WITH_ORDER.format(order=orders[-1]) if orders else FALLBACK
        return Reply(text, [], True, blocked=True, violations=violations, model_answer=model_answer)

    def answer(self, question: str, history: Sequence[tuple[str, str]] = ()) -> Reply:
        client_context = "\n".join(text for role, text in history if role == "client")
        client_text = f"{client_context}\n{question}".strip()
        try:
            model_answer = self.llm.ask(self.system, build_user_message(question, history))
        except LLMError as e:
            log.warning("llm failed: %s", e)
            reply = self.fallback(client_text, [str(e)], None)
            reply.llm_failed = True
            return reply

        result = check(model_answer, self.kb, question, client_context)
        if not result.ok:
            log.warning("guard blocked: %s | %r", result.violations, model_answer.answer)
            return self.fallback(client_text, result.violations, model_answer)

        return Reply(
            text=model_answer.answer.strip(),
            used_fact_ids=model_answer.used_fact_ids,
            needs_operator=model_answer.needs_operator,
            model_answer=model_answer,
        )
