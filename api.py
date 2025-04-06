from __future__ import annotations

import logging
from functools import cache, partial

import slugify as unicode_slug
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from homeassistant.helpers.llm import AssistAPI, LLMContext, Tool
from homeassistant.helpers.template import Template

from .const import CONF_ENTITIES_PROMPT, DEFAULT_ENTITIES_PROMPT

_LOGGER = logging.getLogger(__name__)

LLM_API_FLEX_ASSIST = "flex_assist"
LLM_API_FLEX_ASSIST_NAME = "OmniConv Assist API"


class FlexAssistAPI(AssistAPI):
    """API exposing Assist API to LLMs."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        id_suffix: str = "",
        name_suffix: str = "",
    ) -> None:
        """Init the class."""
        api_id = f"{LLM_API_FLEX_ASSIST}_{id_suffix}" if id_suffix else LLM_API_FLEX_ASSIST
        super(AssistAPI, self).__init__(
            hass=hass,
            id=api_id,
            name=(f"{LLM_API_FLEX_ASSIST_NAME} — {name_suffix}" if name_suffix else LLM_API_FLEX_ASSIST_NAME),
        )
        self.cached_slugify = cache(partial(unicode_slug.slugify, separator="_", lowercase=False))
        self.config_entry = config_entry

    @callback
    def _async_get_exposed_entities_prompt(self, llm_context: LLMContext, exposed_entities: dict | None) -> list[str]:
        """Return the prompt for the API for exposed entities."""
        if not exposed_entities or not exposed_entities.get("entities"):
            return []

        states = [
            state
            for state in self.hass.states.async_all()
            if state.entity_id in exposed_entities.get("entities", {}).keys()
        ]
        entity_registry = er.async_get(self.hass)
        template_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            template_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )

        entities_prompt_template = self.config_entry.options.get(CONF_ENTITIES_PROMPT, DEFAULT_ENTITIES_PROMPT)

        def get_area_name(entity_id):
            area_registry = ar.async_get(self.hass)
            entity_entry = entity_registry.async_get(entity_id)

            if entity_entry and entity_entry.area_id:
                area = area_registry.async_get_area(entity_entry.area_id)
                return area.name if area else None
            return None

        template_vars = {
            "ha_name": self.hass.config.location_name,
            "exposed_entities": template_entities,
            "exposed": exposed_entities,
            "current_device_id": llm_context.device_id,
            "area_name": get_area_name,
        }

        template_obj = Template(entities_prompt_template, self.hass)
        try:
            rendered_prompt = template_obj.async_render(
                template_vars,
                parse_result=False,
                strict=False,  # Prevent UndefinedError for missing variables
            )
        except Exception as err:
            # Fall back to unrendered template if rendering fails
            _LOGGER.error("Error rendering entities prompt template: %s", err)
            rendered_prompt = entities_prompt_template

        return [rendered_prompt]

    @callback
    def _async_get_preable(self, llm_context: LLMContext) -> list[str]:
        """Return the prompt for the API."""

        prompt = []
        if not llm_context.device_id or not async_device_supports_timers(self.hass, llm_context.device_id):
            prompt.append("This device is not able to start timers.")

        return prompt

    @callback
    def _async_get_tools(self, llm_context: LLMContext, exposed_entities: dict | None) -> list[Tool]:
        """Return a list of LLM tools."""

        tools = super()._async_get_tools(llm_context, exposed_entities)
        return [tool for tool in tools if not isinstance(tool, Tool) and not tool.name == "get_home_state"]


async def async_setup_api(hass: HomeAssistant, entry: ConfigEntry) -> None:
    unreg = llm.async_register_api(hass, FlexAssistAPI(hass, entry, entry.entry_id, entry.title))
    entry.async_on_unload(unreg)
