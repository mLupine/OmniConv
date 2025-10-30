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

            all_states = hass.states.async_all()
            exposed_entity_ids = {
                state.entity_id
                for state in all_states
                if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
            }

            exposed_states = [state for state in all_states if state.entity_id in exposed_entity_ids]

            entity_registry = er.async_get(hass)
            exposed_entities = []
            for state in exposed_states:
                entity = entity_registry.async_get(state.entity_id)
                exposed_entities.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.name,
                        "state": state.state,
                        "aliases": entity.aliases if entity and entity.aliases else [],
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
        self._cached_tools: list[CustomFunctionTool] | None = None
        self._cached_yaml_hash: int | None = None

    def _get_custom_functions_as_tools(self) -> list[CustomFunctionTool]:
        """Get custom functions from configuration as Tools."""
        from .exceptions import FunctionNotFound, InvalidFunction

        try:
            function_yaml = self.subentry.data.get(CONF_FUNCTIONS)
            yaml_hash = hash(function_yaml) if function_yaml else 0

            if self._cached_tools is not None and self._cached_yaml_hash == yaml_hash:
                LOGGER.debug("Using cached function tools (%s tools)", len(self._cached_tools))
                return self._cached_tools

            functions = yaml.safe_load(function_yaml) if function_yaml else DEFAULT_CONF_FUNCTIONS

            if not functions:
                self._cached_tools = []
                self._cached_yaml_hash = yaml_hash
                return []

            tools = []
            for function in functions:
                tools.append(
                    CustomFunctionTool(
                        function_spec=function["spec"],
                        function_impl=function["function"],
                    )
                )

            self._cached_tools = tools
            self._cached_yaml_hash = yaml_hash
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
        self._cached_tools = self._get_custom_functions_as_tools()
        LOGGER.debug("Pre-parsed %s custom function tools", len(self._cached_tools))

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

        custom_function_tools = self._cached_tools or self._get_custom_functions_as_tools()
        if custom_function_tools and chat_log.llm_api:
            chat_log.llm_api.tools.extend(custom_function_tools)

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
