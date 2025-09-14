from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from . import ToolExecutor, register_executor

SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required("name"): str,
        vol.Required("description"): str,
        vol.Required("action"): str,
        vol.Optional("parameters", default={}): dict,
    }
)


@register_executor("service")
class ServiceExecutor(ToolExecutor):
    CONFIG_SCHEMA = SERVICE_SCHEMA

    def __init__(self, name: str, description: str, domain: str, service: str, schema: vol.Schema) -> None:
        super().__init__(name, description, schema)
        self._domain = domain
        self._service = service

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ServiceExecutor":
        data = cls.CONFIG_SCHEMA({k: v for k, v in config.items() if k != "type"})
        domain, service = data["action"].split(".")
        schema = vol.Schema(data["parameters"])
        return cls(data["name"], data["description"], domain, service, schema)

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> dict[str, Any]:
        params = self.parameters(tool_input.tool_args)
        await hass.services.async_call(self._domain, self._service, params)
        return {"status": "ok"}
