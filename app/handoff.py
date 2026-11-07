import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_chat_id INTEGER NOT NULL,
    client_name TEXT NOT NULL,
    question TEXT NOT NULL,
    bot_answer TEXT NOT NULL,
    reason TEXT NOT NULL,
    manager_message_id INTEGER,
    status TEXT NOT NULL DEFAULT 'open',
    manager_answer TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS tickets_manager_msg ON tickets(manager_message_id);
"""


@dataclass
class Ticket:
    id: int
    client_chat_id: int
    client_name: str
    question: str
    bot_answer: str
    reason: str
    manager_message_id: int | None
    status: str
    manager_answer: str | None


class TicketStore:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def create(self, client_chat_id: int, client_name: str, question: str, bot_answer: str, reason: str) -> Ticket:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO tickets (client_chat_id, client_name, question, bot_answer, reason) VALUES (?, ?, ?, ?, ?)",
                (client_chat_id, client_name, question, bot_answer, reason),
            )
        return self.get(cur.lastrowid)

    def get(self, ticket_id: int) -> Ticket | None:
        row = self.db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return Ticket(**{k: row[k] for k in Ticket.__dataclass_fields__}) if row else None

    def attach_manager_message(self, ticket_id: int, message_id: int) -> None:
        with self.db:
            self.db.execute("UPDATE tickets SET manager_message_id = ? WHERE id = ?", (message_id, ticket_id))

    def by_manager_message(self, message_id: int) -> Ticket | None:
        row = self.db.execute("SELECT id FROM tickets WHERE manager_message_id = ?", (message_id,)).fetchone()
        return self.get(row["id"]) if row else None

    def answer(self, ticket_id: int, text: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE tickets SET status = 'answered', manager_answer = ? WHERE id = ?", (text, ticket_id)
            )


def handoff_reason(violations: list[str], llm_failed: bool, model_reason: str) -> str:
    if llm_failed:
        return "модель недоступна"
    if violations:
        return "guardrail: " + "; ".join(violations)
    return model_reason or "вопрос вне регламента"


def manager_card(ticket: Ticket) -> str:
    return (
        f"Заявка #{ticket.id}\n"
        f"Клиент: {ticket.client_name}\n\n"
        f"Вопрос: {ticket.question}\n"
        f"Ответ бота: {ticket.bot_answer}\n"
        f"Почему передано: {ticket.reason}\n\n"
        "Ответьте реплаем на это сообщение, ответ уйдёт клиенту."
    )


class DialogHistory:
    def __init__(self, max_turns: int = 3):
        self.turns: dict[int, deque[tuple[str, str]]] = defaultdict(lambda: deque(maxlen=max_turns * 2))

    def get(self, chat_id: int) -> list[tuple[str, str]]:
        return list(self.turns[chat_id])

    def add(self, chat_id: int, question: str, answer: str) -> None:
        self.turns[chat_id].extend([("client", question), ("bot", answer)])
