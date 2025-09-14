from __future__ import annotations

from typing import Any, Callable

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm


class ToolExecutor(llm.Tool):
    def __init__(self, name: str, description: str | None, parameters: vol.Schema) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ToolExecutor":
        raise NotImplementedError

    def bind(self, registry: "ToolRegistry") -> None:
        return


_EXECUTORS: dict[str, type[ToolExecutor]] = {}


def register_executor(name: str) -> Callable[[type[ToolExecutor]], type[ToolExecutor]]:
    def decorator(cls: type[ToolExecutor]) -> type[ToolExecutor]:
        _EXECUTORS[name] = cls
        return cls

    return decorator


class ToolRegistry:
    def __init__(self, executors: list[ToolExecutor]) -> None:
        self._executors = {e.name: e for e in executors}
        for executor in executors:
            executor.bind(self)

    @property
    def tools(self) -> list[llm.Tool]:
        return list(self._executors.values())

    def get(self, name: str) -> ToolExecutor:
        return self._executors[name]


def create_registry(hass: HomeAssistant, configs: list[dict[str, Any]]) -> ToolRegistry:
    executors = [_EXECUTORS[c["type"]].from_config(c) for c in configs]
    return ToolRegistry(executors)


def validate_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_EXECUTORS[c["type"]].CONFIG_SCHEMA({k: v for k, v in c.items() if k != "type"}) for c in configs]


from . import service, script, template, rest, composite  # noqa: E402,F401
