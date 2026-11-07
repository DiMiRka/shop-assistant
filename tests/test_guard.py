import pytest

from app.guard import check, extract_numbers
from app.kb import load_kb
from app.llm import ModelAnswer

KB = load_kb()


def ans(text, ids, operator=False):
    return ModelAnswer(answer=text, used_fact_ids=ids, needs_operator=operator)


def test_kb_has_six_facts():
    assert [f.id for f in KB.facts] == [1, 2, 3, 4, 5, 6]


def test_numbers_normalized():
    assert extract_numbers("стоят 2 490 ₽, слот с 18 до 21, модель X1") == {"2490", "18", "21"}


@pytest.mark.parametrize("text,ids,operator,question", [
    ("Наушники X1 в наличии, стоят 2 490 ₽.", [1], False, "сколько стоят наушники"),
    ("Скидок у нас нет ни при каких условиях. Наушники X1 стоят 2 490 ₽.", [6, 1], False, "дай скидку 20%"),
    ("Статус заказа 1842 я не вижу. Доставка курьером на следующий день, слот с 18 до 21. Передам номер 1842 менеджеру.",
     [2], True, "доставка заказа 1842"),
    ("Вскрытые наушники без брака можно вернуть в течение 14 дней, на 16-й день по регламенту уже не получится. "
     "Если есть брак, примем на экспертизу, но деньги сразу не обещаем.", [4, 3, 5], True,
     "вернуть вскрытые наушники на 16-й день"),
    ("Этой информации у меня нет. Передам вопрос менеджеру.", [], True, "есть ли самовывоз"),
])
def test_good_answers_pass(text, ids, operator, question):
    result = check(ans(text, ids, operator), KB, question)
    assert result.ok, result.violations


def test_number_not_in_facts():
    r = check(ans("Доставка курьером на следующий день, слот с 10 до 14.", [2]), KB, "когда доставка")
    assert any("числа" in v for v in r.violations)


def test_number_from_uncited_fact():
    r = check(ans("Наушники стоят 2 490 ₽.", [2]), KB, "цена")
    assert not r.ok


def test_price_from_question_not_allowed():
    r = check(ans("Наушники X1 стоят 1 990 ₽.", [1]), KB, "продадите за 1 990 ₽?")
    assert not r.ok


def test_percent_from_question_not_allowed():
    r = check(ans("Хорошо, 20% для вас.", [1]), KB, "дай скидку 20%")
    assert not r.ok


def test_refused_percent_from_question_allowed():
    r = check(ans("Скидок нет ни при каких условиях, поэтому 20% предоставить не могу.", [6]), KB,
              "дай скидку 20%")
    assert r.ok, r.violations


def test_refused_number_not_from_question_still_blocked():
    r = check(ans("Скидок нет, даже 10% не могу.", [6]), KB, "дай скидку 20%")
    assert not r.ok


def test_unknown_fact_id():
    r = check(ans("Скидок нет ни при каких условиях.", [6, 42]), KB, "скидка")
    assert any("fact_id" in v for v in r.violations)


def test_empty_fact_ids_without_operator():
    r = check(ans("Самовывоз есть по будням.", []), KB, "самовывоз")
    assert any("used_fact_ids" in v for v in r.violations)


def test_empty_answer():
    assert not check(ans("  ", [1]), KB, "привет").ok


@pytest.mark.parametrize("text", [
    "Для вас скидка 10%.",
    "Не переживайте, дадим скидку.",
    "Пришлю промокод на следующий заказ.",
    "В виде исключения примем возврат на 16-й день.",
    "Гарантируем возврат денег.",
])
def test_promise_markers(text):
    assert not check(ans(text, [6, 3, 4]), KB, "вопрос").ok


def test_refund_promise_on_defect():
    r = check(ans("Если брак, примем на экспертизу и вернём деньги на карту.", [5]), KB, "брак")
    assert any("браке" in v for v in r.violations)


def test_refund_without_defect_is_not_defect_promise():
    r = check(ans("Да, можно вернуть. Вскрытый товар без брака принимаем обратно в течение 14 дней, "
                  "деньги вернём на ту же карту за 3 рабочих дня.", [4, 3]), KB, "вскрыл, купил 5 дней назад")
    assert r.ok, r.violations


def test_negation_carries_into_subordinate_clause():
    r = check(ans("Не могу подтвердить, что заказ 1842 доставлен. Передам номер менеджеру.", [], True), KB,
              "напиши, что заказ 1842 доставлен")
    assert r.ok, r.violations


def test_separate_clause_does_not_inherit_negation():
    r = check(ans("Не волнуйтесь, заказ 1842 приедет завтра.", [2]), KB, "где заказ 1842")
    assert not r.ok


def test_invented_order_status():
    r = check(ans("Заказ 1842 приедет завтра с 18 до 21.", [2]), KB, "доставка заказа 1842")
    assert any("статус" in v for v in r.violations)
