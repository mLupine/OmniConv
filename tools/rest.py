from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client

from . import ToolExecutor, register_executor

REST_SCHEMA = vol.Schema(
    {
        vol.Required("name"): str,
        vol.Required("description"): str,
        vol.Required("url"): str,
        vol.Optional("method", default="get"): str,
        vol.Optional("parameters", default={}): dict,
    }
)


@register_executor("rest")
class RestExecutor(ToolExecutor):
    CONFIG_SCHEMA = REST_SCHEMA

    def __init__(self, name: str, description: str, url: str, method: str, schema: vol.Schema) -> None:
        super().__init__(name, description, schema)
        self._url = url
        self._method = method

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RestExecutor":
        data = cls.CONFIG_SCHEMA({k: v for k, v in config.items() if k != "type"})
        schema = vol.Schema(data["parameters"])
        return cls(data["name"], data["description"], data["url"], data["method"].upper(), schema)

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> dict[str, Any]:
        params = self.parameters(tool_input.tool_args)
        client = get_async_client(hass)
        response = await client.request(self._method, self._url, params=params)
        try:
            return await response.json()
        except Exception:
            return {"status": response.status_code, "text": response.text}
