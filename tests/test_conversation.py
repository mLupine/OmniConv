import types
import importlib
from unittest.mock import AsyncMock, patch

import pytest

from OmniConv.const import CONF_ATTACH_ENTITIES, CONF_PROMPT, DOMAIN
from OmniConv.const import CONF_ENABLE_FUNCTIONS, CONF_FUNCTIONS, CONF_MAX_FUNCTION_CALLS
from homeassistant.helpers import llm
from homeassistant.components.conversation.chat_log import ChatLog, UserContent

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
async def test_prompt_context_attach_false(hass):
    if not CONVERSATION_AVAILABLE:
        pytest.skip("conversation dependencies missing")
    entry = types.SimpleNamespace(subentries={}, runtime_data=None)
    sub = types.SimpleNamespace(
        data={CONF_PROMPT: "t", CONF_ATTACH_ENTITIES: False, "functions": [{"name": "f1"}]},
        title="t",
        subentry_id="1",
        subentry_type="conversation",
    )
    entity = OmniConvConversationEntity(entry, sub)
    entity.hass = hass
    chat_log = types.SimpleNamespace(
        async_provide_llm_data=AsyncMock(),
        unresponded_tool_results=[],
        content=[],
        conversation_id="c1",
    )
    user_input = types.SimpleNamespace(
        as_llm_context=lambda domain: types.SimpleNamespace(
            platform=domain, context=None, language=None, assistant=None, device_id=None
        ),
        extra_system_prompt=None,
        context=types.SimpleNamespace(user_id="u1"),
    )

    class DummyTemplate:
        def __init__(self, template, hass):
            self.template = template

        async def async_render(self, ctx):
            DummyTemplate.context = ctx
            return "rendered"

    hass.auth = types.SimpleNamespace(async_get_user=AsyncMock(return_value=types.SimpleNamespace(name="User")))

    with (
        patch("OmniConv.prompt._get_exposed_entities", return_value={"entities": {}}),
        patch("OmniConv.prompt.template.Template", DummyTemplate),
        patch("homeassistant.components.conversation.async_get_result_from_chat_log", return_value=None),
        patch.object(entity, "_async_handle_chat_log", AsyncMock()),
    ):
        await entity._async_handle_message(user_input, chat_log)
    chat_log.async_provide_llm_data.assert_called_once()
    args = chat_log.async_provide_llm_data.call_args[0]
    assert args[1] is None
    assert args[2] == "rendered"
    ctx = DummyTemplate.context
    assert ctx["exposed_entities"] == {"entities": {}}
    assert ctx["conversation_id"] == "c1"
    assert ctx["user_name"] == "User"
    assert ctx["allowed_functions"] == ["f1"]
    assert ctx["states"] is hass.states


@pytest.mark.asyncio
async def test_prompt_context_attach_true(hass):
    if not CONVERSATION_AVAILABLE:
        pytest.skip("conversation dependencies missing")
    entry = types.SimpleNamespace(subentries={}, runtime_data=None)
    from homeassistant.const import CONF_LLM_HASS_API

    llm_api = object()
    sub = types.SimpleNamespace(
        data={
            CONF_PROMPT: "t",
            CONF_ATTACH_ENTITIES: True,
            CONF_LLM_HASS_API: llm_api,
            "functions": [{"name": "f1"}],
        },
        title="t",
        subentry_id="1",
        subentry_type="conversation",
    )
    entity = OmniConvConversationEntity(entry, sub)
    entity.hass = hass
    chat_log = types.SimpleNamespace(
        async_provide_llm_data=AsyncMock(),
        unresponded_tool_results=[],
        content=[],
        conversation_id="c2",
    )
    user_input = types.SimpleNamespace(
        as_llm_context=lambda domain: types.SimpleNamespace(
            platform=domain, context=None, language=None, assistant=None, device_id=None
        ),
        extra_system_prompt=None,
        context=types.SimpleNamespace(user_id="u2"),
    )

    class DummyTemplate:
        def __init__(self, template, hass):
            self.template = template

        async def async_render(self, ctx):
            DummyTemplate.context = ctx
            return "rendered"

    hass.auth = types.SimpleNamespace(async_get_user=AsyncMock(return_value=types.SimpleNamespace(name="User2")))

    with (
        patch("OmniConv.prompt._get_exposed_entities", return_value={"entities": {}}),
        patch("OmniConv.prompt.template.Template", DummyTemplate),
        patch("homeassistant.components.conversation.async_get_result_from_chat_log", return_value=None),
        patch.object(entity, "_async_handle_chat_log", AsyncMock()),
    ):
        await entity._async_handle_message(user_input, chat_log)
    chat_log.async_provide_llm_data.assert_called_once()
    args = chat_log.async_provide_llm_data.call_args[0]
    assert args[1] is llm_api
    assert args[2] == "rendered"
    ctx = DummyTemplate.context
    assert ctx["conversation_id"] == "c2"
    assert ctx["user_name"] == "User2"
    assert ctx["allowed_functions"] == ["f1"]


@pytest.mark.asyncio
async def test_finish_event_fired(hass):
    if not CONVERSATION_AVAILABLE:
        pytest.skip("conversation dependencies missing")
    entry = types.SimpleNamespace(
        subentries={},
        runtime_data=types.SimpleNamespace(
            responses=types.SimpleNamespace(
                create=AsyncMock(
                    return_value=types.SimpleNamespace(response=types.SimpleNamespace(model_dump=lambda: {}))
                )
            )
        ),
    )
    sub = types.SimpleNamespace(data={}, title="t", subentry_id="1", subentry_type="conversation")
    entity = OmniConvConversationEntity(entry, sub)
    entity.hass = hass
    chat_log = ChatLog(hass, "c1", [UserContent("hi")])

    def fake_add_delta_content_stream(self, entity_id, agen):
        async def gen():
            async for _ in agen:
                pass
            if False:
                yield None

        return gen()

    chat_log.async_add_delta_content_stream = types.MethodType(fake_add_delta_content_stream, chat_log)

    async def empty_gen():
        if False:
            yield None

    events: list = []
    hass.bus.async_listen(f"{DOMAIN}.conversation.finished", lambda e: events.append(e.data))
    with patch("OmniConv.entity._transform_stream", return_value=empty_gen()):
        await entity._async_handle_chat_log(chat_log)
    assert events[0]["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_service_function_called(hass):
    if not CONVERSATION_AVAILABLE:
        pytest.skip("conversation dependencies missing")

    calls: list = []

    async def handler(call):
        calls.append(call.data)

    hass.services.async_register("light", "turn_on", handler)

    svc_cfg = {
        "type": "service",
        "name": "svc",
        "description": "d",
        "action": "light.turn_on",
        "parameters": {"entity_id": str},
    }

    entry = types.SimpleNamespace(
        subentries={},
        runtime_data=types.SimpleNamespace(
            responses=types.SimpleNamespace(
                create=AsyncMock(
                    return_value=types.SimpleNamespace(response=types.SimpleNamespace(model_dump=lambda: {}))
                )
            )
        ),
    )
    sub = types.SimpleNamespace(
        data={
            CONF_ENABLE_FUNCTIONS: True,
            CONF_FUNCTIONS: [svc_cfg],
            CONF_MAX_FUNCTION_CALLS: 1,
        },
        title="t",
        subentry_id="1",
        subentry_type="conversation",
    )
    entity = OmniConvConversationEntity(entry, sub)
    entity.hass = hass
    chat_log = ChatLog(hass, "c1", [UserContent("hi")])

    async def fake_transform(chat_log, stream):
        yield {
            "role": "assistant",
            "tool_calls": [llm.ToolInput(id="1", tool_name="svc", tool_args={"entity_id": "light.k"})],
        }

    with patch("OmniConv.entity._transform_stream", fake_transform):
        await entity._async_handle_chat_log(chat_log)

    assert calls[0]["entity_id"] == "light.k"


@pytest.mark.asyncio
async def test_default_prompt(hass):
    if not CONVERSATION_AVAILABLE:
        pytest.skip("conversation dependencies missing")
    entry = types.SimpleNamespace(subentries={}, runtime_data=None)
    sub = types.SimpleNamespace(
        data={CONF_ATTACH_ENTITIES: True},
        title="t",
        subentry_id="1",
        subentry_type="conversation",
    )
    entity = OmniConvConversationEntity(entry, sub)
    entity.hass = hass
    chat_log = types.SimpleNamespace(
        async_provide_llm_data=AsyncMock(),
        unresponded_tool_results=[],
        content=[],
        conversation_id="c3",
    )
    user_input = types.SimpleNamespace(
        as_llm_context=lambda domain: types.SimpleNamespace(
            platform=domain, context=None, language=None, assistant=None, device_id=None
        ),
        extra_system_prompt=None,
        context=None,
    )

    class DummyTemplate:
        def __init__(self, template, hass):
            DummyTemplate.template = template

        async def async_render(self, ctx):
            return "rendered"

    from homeassistant.helpers import llm

    with (
        patch("OmniConv.prompt._get_exposed_entities", return_value={}),
        patch("OmniConv.prompt.template.Template", DummyTemplate),
        patch("homeassistant.components.conversation.async_get_result_from_chat_log", return_value=None),
        patch.object(entity, "_async_handle_chat_log", AsyncMock()),
    ):
        await entity._async_handle_message(user_input, chat_log)
    assert DummyTemplate.template == llm.DEFAULT_INSTRUCTIONS_PROMPT
