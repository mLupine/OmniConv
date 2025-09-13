import pathlib
import sys
import types
import importlib
from unittest.mock import AsyncMock, patch

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from OmniConv.const import CONF_ATTACH_ENTITIES, CONF_PROMPT

hce = importlib.import_module("homeassistant.config_entries")
if not hasattr(hce, "ConfigSubentry"):

    class ConfigSubentry:  # type: ignore[no-redef]
        def __init__(self, **data):
            self.__dict__.update(data)

    setattr(hce, "ConfigSubentry", ConfigSubentry)

try:
    from OmniConv.conversation import OmniConvConversationEntity

    CONVERSATION_AVAILABLE = True
except Exception:  # pragma: no cover
    CONVERSATION_AVAILABLE = False


@pytest.mark.asyncio
async def test_attach_entities_false(hass):
    if not CONVERSATION_AVAILABLE:
        pytest.skip("conversation dependencies missing")
    entry = types.SimpleNamespace(subentries={}, runtime_data=None)
    sub = types.SimpleNamespace(
        data={CONF_PROMPT: "hi", CONF_ATTACH_ENTITIES: False},
        title="t",
        subentry_id="1",
        subentry_type="conversation",
    )
    entity = OmniConvConversationEntity(entry, sub)
    chat_log = types.SimpleNamespace(
        async_provide_llm_data=AsyncMock(),
        unresponded_tool_results=[],
        content=[],
    )
    user_input = types.SimpleNamespace(
        as_llm_context=lambda domain: types.SimpleNamespace(
            platform=domain, context=None, language=None, assistant=None, device_id=None
        ),
        extra_system_prompt=None,
    )

    class DummyTemplate:
        def __init__(self, *args, **kwargs):
            pass

        def async_render(self, *args, **kwargs):
            return ""

    with (
        patch("OmniConv.conversation._get_exposed_entities", return_value={"entities": {}}),
        patch("OmniConv.conversation.template.Template", DummyTemplate),
        patch("homeassistant.components.conversation.async_get_result_from_chat_log", return_value=None),
        patch.object(entity, "_async_handle_chat_log", AsyncMock()),
    ):
        await entity._async_handle_message(user_input, chat_log)
    chat_log.async_provide_llm_data.assert_called_once()
    args = chat_log.async_provide_llm_data.call_args[0]
    assert args[1] is None
