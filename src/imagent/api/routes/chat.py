"""Conversacion."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter

from imagent.api.deps import ConversationDep
from imagent.api.schemas import ChatIn, ChatOut
from imagent.observability.logging import log_extra

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("")
async def preguntar(peticion: ChatIn, conversacion: ConversationDep) -> ChatOut:
    """Un turno.

    El `thread_id` lo pone el cliente o se genera aqui. Es la unica clave del
    estado persistido: mandar el mismo continua la conversacion, mandar otro
    empieza una nueva. No hay nada mas que guardar en el cliente.
    """
    thread_id = peticion.thread_id or str(uuid4())

    resultado = await conversacion.ask(thread_id=thread_id, question=peticion.message)

    logger.info(
        "turno respondido",
        extra=log_extra(
            thread_id=thread_id,
            incomplete=resultado.incomplete,
            reason=resultado.incomplete_reason or "-",
            degradations=len(resultado.degradations),
        ),
    )
    return ChatOut.de(thread_id, resultado)
