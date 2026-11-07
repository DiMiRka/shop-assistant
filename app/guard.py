import re
from dataclasses import dataclass, field

from app.kb import KnowledgeBase
from app.llm import ModelAnswer

NUMBER_RE = re.compile(r"(?<![A-Za-zА-Яа-яЁё\d])(\d{1,3}(?:[   ]\d{3})+(?!\d)|\d+(?:[.,]\d+)?)")
MONEY_OR_PERCENT_AFTER = re.compile(r"^\s*(%|₽|руб|р\.|процент)", re.IGNORECASE)
CLAUSE_SPLIT = re.compile(r"[.!?;,:\n—–]|\s-\s")
SENTENCE_SPLIT = re.compile(r"[.!?\n]")
NEGATION = re.compile(r"\b(не|нет|ни|нельзя|невозможно)\b", re.IGNORECASE)

PROMISE_MARKERS = re.compile(r"скидк|промокод|акци[яиюейя]|исключени|бонус|компенсац|гарант|подар", re.IGNORECASE)
DEFECT_CONTEXT = re.compile(r"(?<!без )брак|экспертиз", re.IGNORECASE)
SUBORDINATE = re.compile(r"^(что|чтобы|будто|ли)\b", re.IGNORECASE)
MONEY_WORDS = re.compile(r"деньг|денег|средств|стоимост", re.IGNORECASE)
REFUND_VERBS = re.compile(r"верн|возвра|получ|прид|перевед|компенс|выплат", re.IGNORECASE)
ORDER_STATUS = re.compile(
    r"приед|привез|доставлен|доставим|отправлен|в пути|передан курьер|собран|прибуд|получите|сегодня|завтра|уже",
    re.IGNORECASE,
)
DELIVERY_CLAIM = re.compile(
    r"приед|привез|доставлен|отправлен|в пути|передан курьер|собран|прибуд|сегодня|завтра", re.IGNORECASE
)
ORDER_REF = re.compile(r"заказ\w*\s*(?:№|#|номер)?\s*(\d+)", re.IGNORECASE)


@dataclass
class GuardResult:
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def normalize_number(raw: str) -> str:
    return re.sub(r"[   ]", "", raw).replace(",", ".")


def extract_numbers(text: str) -> set[str]:
    return {normalize_number(m.group(1)) for m in NUMBER_RE.finditer(text)}


def neutral_question_numbers(question: str) -> set[str]:
    result = set()
    for m in NUMBER_RE.finditer(question):
        if not MONEY_OR_PERCENT_AFTER.match(question[m.end():]):
            result.add(normalize_number(m.group(1)))
    return result


def order_numbers(question: str) -> set[str]:
    found = {m.group(1) for m in ORDER_REF.finditer(question)}
    found |= {n for n in extract_numbers(question) if len(n) >= 4 and n.isdigit()}
    return found


def clauses(text: str) -> list[str]:
    result: list[str] = []
    for part in (c.strip() for c in CLAUSE_SPLIT.split(text)):
        if not part:
            continue
        if result and SUBORDINATE.match(part):
            result[-1] = f"{result[-1]}, {part}"
        else:
            result.append(part)
    return result


def check(answer: ModelAnswer, kb: KnowledgeBase, question: str, client_context: str = "") -> GuardResult:
    result = GuardResult()
    question = f"{client_context}\n{question}".strip()
    text = answer.answer.strip()

    if not text:
        result.violations.append("пустой ответ")
        return result

    unknown = [i for i in answer.used_fact_ids if kb.get(i) is None]
    if unknown:
        result.violations.append(f"несуществующие fact_id: {unknown}")

    if not answer.used_fact_ids and not answer.needs_operator:
        result.violations.append("нет used_fact_ids и нет передачи оператору")

    cited = [kb.get(i) for i in answer.used_fact_ids if kb.get(i) is not None]
    allowed = set().union(*(extract_numbers(f.fact) for f in cited)) | neutral_question_numbers(question)
    refused = {n for c in clauses(text) if NEGATION.search(c) for n in extract_numbers(c)}
    extra = sorted(extract_numbers(text) - allowed - (refused & extract_numbers(question)))
    if extra:
        result.violations.append(f"числа не из процитированных фактов: {extra}")

    for clause in clauses(text):
        if PROMISE_MARKERS.search(clause) and not NEGATION.search(clause):
            result.violations.append(f"обещание вне регламента: «{clause}»")

    for sentence in SENTENCE_SPLIT.split(text):
        if not DEFECT_CONTEXT.search(sentence):
            continue
        for clause in clauses(sentence):
            if MONEY_WORDS.search(clause) and REFUND_VERBS.search(clause) and not NEGATION.search(clause):
                result.violations.append(f"обещание денег при браке: «{clause}»")

    orders = order_numbers(question)
    about_order = bool(orders) or "заказ" in question.lower()
    for clause in clauses(text):
        if NEGATION.search(clause):
            continue
        mentions_order = "заказ" in clause.lower() or any(n in extract_numbers(clause) for n in orders)
        if (mentions_order and ORDER_STATUS.search(clause)) or (about_order and DELIVERY_CLAIM.search(clause)):
            result.violations.append(f"выдуманный статус заказа: «{clause}»")

    return result
