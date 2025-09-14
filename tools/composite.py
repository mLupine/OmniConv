from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from . import ToolExecutor, register_executor, ToolRegistry

COMPOSITE_SCHEMA = vol.Schema(
    {
        vol.Required("name"): str,
        vol.Required("description"): str,
        vol.Required("sequence"): [str],
        vol.Optional("parameters", default={}): dict,
    }
)


@register_executor("composite")
class CompositeExecutor(ToolExecutor):
    CONFIG_SCHEMA = COMPOSITE_SCHEMA

    def __init__(self, name: str, description: str, sequence: list[str], schema: vol.Schema) -> None:
        super().__init__(name, description, schema)
        self._sequence = sequence
        self._registry: ToolRegistry | None = None

    def bind(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompositeExecutor":
        data = cls.CONFIG_SCHEMA({k: v for k, v in config.items() if k != "type"})
        schema = vol.Schema(data["parameters"])
        return cls(data["name"], data["description"], data["sequence"], schema)

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> dict[str, Any]:
        if self._registry is None:
            raise ValueError("Registry not bound")
        params = self.parameters(tool_input.tool_args)
        results: list[Any] = []
        for name in self._sequence:
            executor = self._registry.get(name)
            call_input = llm.ToolInput(id=name, tool_name=name, tool_args=params)
            results.append(await executor.async_call(hass, call_input, llm_context))
        return {"results": results}
