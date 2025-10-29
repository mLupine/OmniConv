"""Conversation support for OmniConv."""

import yaml
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OmniConvConfigEntry
from .const import (
    CONF_FUNCTIONS,
    CONF_LLM_HASS_API,
    CONF_PROMPT,
    DEFAULT_CONF_FUNCTIONS,
    DOMAIN,
    LOGGER,
)
from .entity import OmniConvBaseLLMEntity


class CustomFunctionTool(llm.Tool):
    """Tool for executing custom functions defined in the configuration."""

    def __init__(self, function_spec: dict, function_impl: dict) -> None:
        """Initialize the tool with function specification and implementation."""
        self.name = function_spec["name"]
        self.description = function_spec.get("description", f"Execute {self.name} function")
        self.parameters = {}
        self.function_impl = function_impl
        self.function_spec = function_spec

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Execute the function when called as a tool."""
        from homeassistant.components.homeassistant.exposed_entities import async_should_expose
        from homeassistant.helpers import entity_registry as er

        from .helpers import get_function_executor

        try:
            function_executor = get_function_executor(self.function_impl["type"])

            exposed_entities = []
            states = [
                state
                for state in hass.states.async_all()
                if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
            ]
            entity_registry = er.async_get(hass)
            for state in states:
                entity_id = state.entity_id
                entity = entity_registry.async_get(entity_id)
                aliases = []
                if entity and entity.aliases:
                    aliases = entity.aliases
                exposed_entities.append(
                    {
                        "entity_id": entity_id,
                        "name": state.name,
                        "state": hass.states.get(entity_id).state,
                        "aliases": aliases,
                    }
                )

            user_input = conversation.ConversationInput(
                text=llm_context.user_prompt or "",
                conversation_id=tool_input.id,
                language=llm_context.language,
                context=llm_context.context,
                device_id=llm_context.device_id,
                agent_id=llm_context.assistant,
            )

            result = await function_executor.execute(
                hass,
                self.function_impl,
                tool_input.tool_args,
                user_input,
                exposed_entities,
            )

            LOGGER.info(
                "Custom function %s executed successfully with result: %s",
                self.name,
                result,
            )
            return result

        except Exception as err:
            LOGGER.error("Error executing function %s: %s", self.name, err)
            return {"error": str(err)}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OmniConvConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [OmniConvEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class OmniConvEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    OmniConvBaseLLMEntity,
):
    """OmniConv conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, entry: OmniConvConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def _get_custom_functions_as_tools(self) -> list[CustomFunctionTool]:
        """Get custom functions from configuration as Tools."""
        from .exceptions import FunctionNotFound, InvalidFunction

        try:
            function_yaml = self.subentry.data.get(CONF_FUNCTIONS)
            functions = yaml.safe_load(function_yaml) if function_yaml else DEFAULT_CONF_FUNCTIONS

            if not functions:
                return []

            tools = []
            for function in functions:
                tools.append(
                    CustomFunctionTool(
                        function_spec=function["spec"],
                        function_impl=function["function"],
                    )
                )

            LOGGER.info("Created %s custom function tools", len(tools))
            return tools

        except (InvalidFunction, FunctionNotFound) as err:
            LOGGER.error("Error loading functions: %s", err)
            return []
        except Exception as err:
            LOGGER.error("Unexpected error loading functions: %s", err)
            return []

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the API."""
        options = self.subentry.data

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        custom_function_tools = self._get_custom_functions_as_tools()
        if custom_function_tools and chat_log.llm_api:
            for tool in custom_function_tools:
                chat_log.llm_api.tools.append(tool)

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
