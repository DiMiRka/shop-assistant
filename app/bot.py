import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from app.assistant import Assistant
from app.handoff import DialogHistory, TicketStore, handoff_reason, manager_card
from app.llm import provider_from_env

log = logging.getLogger("bot")


def client_name(message: Message) -> str:
    user = message.from_user
    if user is None:
        return f"chat {message.chat.id}"
    name = user.full_name + (f" (@{user.username})" if user.username else "")
    return f"{name}, chat {message.chat.id}"


def build_router(assistant: Assistant, tickets: TicketStore, history: DialogHistory,
                 manager_chat_id: int | None) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer("Здравствуйте! Я помощник магазина, чем могу помочь?")

    @router.message(Command("chatid"))
    async def chat_id(message: Message) -> None:
        await message.answer(f"chat_id: {message.chat.id}")

    if manager_chat_id is not None:
        @router.message(F.chat.id == manager_chat_id, F.reply_to_message)
        async def manager_reply(message: Message, bot: Bot) -> None:
            ticket = tickets.by_manager_message(message.reply_to_message.message_id)
            if ticket is None:
                await message.reply("Это не заявка. Ответьте реплаем на карточку заявки.")
                return
            text = message.text or message.caption or ""
            if message.text:
                await bot.send_message(ticket.client_chat_id, f"Ответ менеджера:\n{text}")
            else:
                await bot.send_message(ticket.client_chat_id, "Ответ менеджера:")
                await message.copy_to(ticket.client_chat_id)
            tickets.answer(ticket.id, text)
            history.add(ticket.client_chat_id, ticket.question, f"Менеджер: {text}")
            await message.reply(f"Отправлено клиенту, заявка #{ticket.id} закрыта.")

        @router.message(F.chat.id == manager_chat_id)
        async def manager_other(message: Message) -> None:
            return

    @router.message(F.text)
    async def handle(message: Message, bot: Bot) -> None:
        reply = await asyncio.to_thread(assistant.answer, message.text, history.get(message.chat.id))
        log.info("q=%r facts=%s operator=%s blocked=%s violations=%s",
                 message.text, reply.used_fact_ids, reply.needs_operator, reply.blocked, reply.violations)
        await message.answer(reply.text)
        history.add(message.chat.id, message.text, reply.text)

        if not reply.needs_operator:
            return
        reason = handoff_reason(reply.violations, reply.llm_failed,
                                reply.model_answer.reason if reply.model_answer else "")
        ticket = tickets.create(message.chat.id, client_name(message), message.text, reply.text, reason)
        if manager_chat_id is None:
            log.warning("ticket #%s created, MANAGER_CHAT_ID not set", ticket.id)
            return
        card = await bot.send_message(manager_chat_id, manager_card(ticket))
        tickets.attach_manager_message(ticket.id, card.message_id)

    return router


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    manager = os.getenv("MANAGER_CHAT_ID")
    dp = Dispatcher()
    dp.include_router(build_router(
        Assistant(provider_from_env()),
        TicketStore(os.getenv("DB_PATH", "data/tickets.db")),
        DialogHistory(),
        int(manager) if manager else None,
    ))
    await dp.start_polling(Bot(os.environ["TELEGRAM_BOT_TOKEN"]))


if __name__ == "__main__":
    asyncio.run(main())
