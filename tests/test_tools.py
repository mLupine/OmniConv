import types
from unittest.mock import AsyncMock

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from OmniConv import tools


@pytest.mark.asyncio
async def test_service_executor_calls_service(hass: HomeAssistant):
    calls: list = []

    async def handler(call):
        calls.append(call.data)

    hass.services.async_register("light", "turn_on", handler)

    cfg = {
        "type": "service",
        "name": "turn_on_light",
        "description": "Turn on",
        "action": "light.turn_on",
        "parameters": {"entity_id": str},
    }
    registry = tools.create_registry(hass, [cfg])
    tool = registry.get("turn_on_light")
    tool_input = llm.ToolInput(id="1", tool_name="turn_on_light", tool_args={"entity_id": "light.k"})
    await tool.async_call(hass, tool_input, types.SimpleNamespace())
    assert calls[0]["entity_id"] == "light.k"


@pytest.mark.asyncio
async def test_template_executor_renders(hass: HomeAssistant):
    cfg = {
        "type": "template",
        "name": "greet",
        "description": "Greet",
        "template": "Hello {{ name }}",
        "parameters": {"name": str},
    }
    registry = tools.create_registry(hass, [cfg])
    tool = registry.get("greet")
    tool_input = llm.ToolInput(id="1", tool_name="greet", tool_args={"name": "Bob"})
    result = await tool.async_call(hass, tool_input, types.SimpleNamespace())
    assert result["result"] == "Hello Bob"


@pytest.mark.asyncio
async def test_rest_executor(hass: HomeAssistant, monkeypatch):
    cfg = {
        "type": "rest",
        "name": "fetch",
        "description": "Fetch",
        "url": "http://example/test",
    }
    client = AsyncMock()
    client.request.return_value.json = AsyncMock(return_value={"a": 1})
    monkeypatch.setattr("OmniConv.tools.rest.get_async_client", lambda hass: client)
    registry = tools.create_registry(hass, [cfg])
    tool = registry.get("fetch")
    tool_input = llm.ToolInput(id="1", tool_name="fetch", tool_args={})
    result = await tool.async_call(hass, tool_input, types.SimpleNamespace())
    assert result == {"a": 1}


@pytest.mark.asyncio
async def test_composite_executor(hass: HomeAssistant):
    service_cfg = {
        "type": "service",
        "name": "svc",
        "description": "svc",
        "action": "light.turn_on",
    }
    template_cfg = {
        "type": "template",
        "name": "tmp",
        "description": "tmp",
        "template": "done",
    }
    composite_cfg = {
        "type": "composite",
        "name": "combo",
        "description": "combo",
        "sequence": ["tmp"],
    }
    calls: list = []

    async def handler(call):
        calls.append(1)

    hass.services.async_register("light", "turn_on", handler)

    registry = tools.create_registry(hass, [service_cfg, template_cfg, composite_cfg])
    tool = registry.get("combo")
    tool_input = llm.ToolInput(id="1", tool_name="combo", tool_args={})
    result = await tool.async_call(hass, tool_input, types.SimpleNamespace())
    assert result["results"][0]["result"] == "done"
    assert calls == []
