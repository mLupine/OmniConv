from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from . import ToolExecutor, register_executor

SCRIPT_SCHEMA = vol.Schema(
    {
        vol.Required("name"): str,
        vol.Required("description"): str,
        vol.Required("script"): str,
        vol.Optional("parameters", default={}): dict,
    }
)


@register_executor("script")
class ScriptExecutor(ToolExecutor):
    CONFIG_SCHEMA = SCRIPT_SCHEMA

    def __init__(self, name: str, description: str, script: str, schema: vol.Schema) -> None:
        super().__init__(name, description, schema)
        self._script = script

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ScriptExecutor":
        data = cls.CONFIG_SCHEMA({k: v for k, v in config.items() if k != "type"})
        schema = vol.Schema(data["parameters"])
        return cls(data["name"], data["description"], data["script"], schema)

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> dict[str, Any]:
        params = self.parameters(tool_input.tool_args)
        await hass.services.async_call("script", self._script, params)
        return {"status": "ok"}
