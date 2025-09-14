from __future__ import annotations

from typing import Any
from inspect import isawaitable

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, template
from homeassistant.helpers.llm import _get_exposed_entities

from .const import CONF_FUNCTIONS, CONF_PROMPT, DOMAIN


async def async_render_prompt(
    hass: HomeAssistant, options: dict[str, Any], chat_log: conversation.ChatLog, user_name: str | None
) -> str:
    """Render the prompt template."""
    prompt = options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)
    tmpl = template.Template(prompt, hass)
    context: dict[str, Any] = {
        "exposed_entities": _get_exposed_entities(hass, DOMAIN),
        "conversation_id": chat_log.conversation_id,
        "user_name": user_name,
        "allowed_functions": [f["name"] for f in options.get(CONF_FUNCTIONS, [])],
        "states": hass.states,
    }
    rendered = tmpl.async_render(context)
    if isawaitable(rendered):
        return await rendered
    return rendered
