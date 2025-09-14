from __future__ import annotations

from typing import Any
from inspect import isawaitable

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, template

from . import ToolExecutor, register_executor

TEMPLATE_SCHEMA = vol.Schema(
    {
        vol.Required("name"): str,
        vol.Required("description"): str,
        vol.Required("template"): str,
        vol.Optional("parameters", default={}): dict,
    }
)


@register_executor("template")
class TemplateExecutor(ToolExecutor):
    CONFIG_SCHEMA = TEMPLATE_SCHEMA

    def __init__(self, name: str, description: str, template_str: str, schema: vol.Schema) -> None:
        super().__init__(name, description, schema)
        self._template = template_str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TemplateExecutor":
        data = cls.CONFIG_SCHEMA({k: v for k, v in config.items() if k != "type"})
        schema = vol.Schema(data["parameters"])
        return cls(data["name"], data["description"], data["template"], schema)

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> dict[str, Any]:
        params = self.parameters(tool_input.tool_args)
        tpl = template.Template(self._template, hass)
        result = tpl.async_render(params)
        if isawaitable(result):
            result = await result
        return {"result": result}
