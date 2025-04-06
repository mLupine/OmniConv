"""Conversation support for OmniConv."""

import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal

import openai
import voluptuous as vol
import yaml
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from openai._streaming import AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ToolParam,
    WebSearchToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.web_search_tool_param import UserLocation
from voluptuous_openapi import convert

from . import OmniConvConfigEntry
from .api import LLM_API_FLEX_ASSIST
from .const import (
    CONF_CHAT_MODEL,
    CONF_FUNCTIONS,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITY,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    CONF_WEB_SEARCH_COUNTRY,
    CONF_WEB_SEARCH_REGION,
    CONF_WEB_SEARCH_TIMEZONE,
    CONF_WEB_SEARCH_USER_LOCATION,
    DEFAULT_CONF_FUNCTIONS,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
)
from .helpers import (
    is_azure,
    log_openai_request,
    log_openai_response,
    log_openai_stream_event,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


class CustomFunctionTool(llm.Tool):
    """Tool for executing custom functions defined in the configuration."""

    def __init__(self, function_spec: dict, function_impl: dict) -> None:
        """Initialize the tool with function specification and implementation."""
        self.name = function_spec["name"]
        self.description = function_spec.get("description", f"Execute {self.name} function")

        self.parameters = vol.Schema({})

        self.function_impl = function_impl
        self.function_spec = function_spec

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.JsonObjectType:
        """Execute the function when called as a tool."""
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

            from homeassistant.components.conversation import ConversationInput

            user_input = ConversationInput(
                text=llm_context.user_prompt or "",
                conversation_id=tool_input.id,
                language=llm_context.language,
                context=llm_context.context,
                device_id=llm_context.device_id,
                agent_id=llm_context.assistant,  # Use the assistant name as agent_id
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
    agent = OmniConvEntity(config_entry)
    async_add_entities([agent])


def _format_tool(tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None) -> FunctionToolParam:
    """Format tool specification."""
    if isinstance(tool, CustomFunctionTool):
        return FunctionToolParam(
            type="function",
            name=tool.name,
            parameters=tool.function_spec.get("parameters", {}),
            description=tool.description,
            strict=False,
        )
    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        description=tool.description,
        strict=False,
    )


def _convert_content_to_param(
    content: conversation.Content,
) -> ResponseInputParam:
    """Convert any native chat message for this agent to the native format."""
    messages: ResponseInputParam = []
    if isinstance(content, conversation.ToolResultContent):
        return [
            FunctionCallOutput(
                type="function_call_output",
                call_id=content.tool_call_id,
                output=json.dumps(content.tool_result),
            )
        ]

    if content.content:
        role: Literal["user", "assistant", "system", "developer"] = content.role
        if role == "system":
            role = "developer"
        messages.append(EasyInputMessageParam(type="message", role=role, content=content.content))

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        messages.extend(
            ResponseFunctionToolCallParam(
                type="function_call",
                name=tool_call.tool_name,
                arguments=json.dumps(tool_call.tool_args),
                call_id=tool_call.id,
            )
            for tool_call in content.tool_calls
        )
    return messages


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncStream[ResponseStreamEvent],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    async for event in result:
        log_openai_stream_event(event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseOutputMessage):
                yield {"role": event.item.role}
            elif isinstance(event.item, ResponseFunctionToolCall):
                current_tool_call = event.item
        elif isinstance(event, ResponseTextDeltaEvent):
            yield {"content": event.delta}
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            current_tool_call.status = "completed"
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call.call_id,
                        tool_name=current_tool_call.name,
                        tool_args=json.loads(current_tool_call.arguments),
                    )
                ]
            }
        elif isinstance(event, ResponseCompletedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
        elif isinstance(event, ResponseIncompleteEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )

            if event.response.incomplete_details and event.response.incomplete_details.reason:
                reason: str = event.response.incomplete_details.reason
            else:
                reason = "unknown reason"

            if reason == "max_output_tokens":
                reason = "max output tokens reached"
            elif reason == "content_filter":
                reason = "content filter triggered"

            raise HomeAssistantError(f"OpenAI response incomplete: {reason}")
        elif isinstance(event, ResponseFailedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
            reason = "unknown reason"
            if event.response.error is not None:
                reason = event.response.error.message
            raise HomeAssistantError(f"OpenAI response failed: {reason}")
        elif isinstance(event, ResponseErrorEvent):
            raise HomeAssistantError(f"OpenAI response error: {event.message}")


class OmniConvEntity(conversation.ConversationEntity, conversation.AbstractConversationAgent):
    """OmniConv conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OmniConvConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenAI",
            model="ChatGPT",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def _get_custom_functions_as_tools(self) -> list[CustomFunctionTool]:
        """Get custom functions from configuration as Tools."""
        from .exceptions import FunctionNotFound, InvalidFunction

        try:
            function_yaml = self.entry.options.get(CONF_FUNCTIONS)
            functions = yaml.safe_load(function_yaml) if function_yaml else DEFAULT_CONF_FUNCTIONS

            if not functions:
                return []

            tools = []
            for function in functions:
                # Create a CustomFunctionTool for each function with both spec and implementation
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
        assist_pipeline.async_migrate_engine(self.hass, "conversation", self.entry.entry_id, self.entity_id)
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(self.entry.add_update_listener(self._async_entry_update_listener))

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options

        try:
            llm_api_id = f"{LLM_API_FLEX_ASSIST}_{self.entry.entry_id}"
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                llm_api_id,
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[ToolParam] | None = None
        if chat_log.llm_api:
            tools = [_format_tool(tool, chat_log.llm_api.custom_serializer) for tool in chat_log.llm_api.tools]

        # Add custom functions as tools
        custom_function_tools = self._get_custom_functions_as_tools()
        if custom_function_tools:
            if tools is None:
                tools = []

            if chat_log.llm_api:
                for tool in custom_function_tools:
                    chat_log.llm_api.tools.append(tool)

            for tool in custom_function_tools:
                tools.append(
                    _format_tool(
                        tool,
                        (chat_log.llm_api.custom_serializer if chat_log.llm_api else None),
                    )
                )
            LOGGER.info(
                "Added %s custom function tools to the conversation",
                len(custom_function_tools),
            )

        if options.get(CONF_WEB_SEARCH):
            web_search = WebSearchToolParam(
                type="web_search_preview",
                search_context_size=options.get(CONF_WEB_SEARCH_CONTEXT_SIZE, RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE),
            )
            if options.get(CONF_WEB_SEARCH_USER_LOCATION):
                web_search["user_location"] = UserLocation(
                    type="approximate",
                    city=options.get(CONF_WEB_SEARCH_CITY, ""),
                    region=options.get(CONF_WEB_SEARCH_REGION, ""),
                    country=options.get(CONF_WEB_SEARCH_COUNTRY, ""),
                    timezone=options.get(CONF_WEB_SEARCH_TIMEZONE, ""),
                )
            if tools is None:
                tools = []
            tools.append(web_search)

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        messages = [m for content in chat_log.content for m in _convert_content_to_param(content)]

        client = self.entry.runtime_data

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": model,
                "input": messages,
                "max_output_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "store": False,
                "stream": True,
            }
            if tools:
                model_args["tools"] = tools

            if model.startswith("o"):
                model_args["reasoning"] = {"effort": options.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)}

            try:
                # Handle Azure OpenAI if applicable
                if is_azure(self.entry.options.get("base_url", "")):
                    # Azure OpenAI uses deployments instead of model names
                    # So we need to adjust the model parameter
                    deployment = model
                    # Azure uses 'deployment' parameter instead of 'model'
                    model_args.pop("model")
                    model_args["deployment"] = deployment

                # Log the request parameters
                log_openai_request("responses.create", **model_args)

                # Make the API call
                result = await client.responses.create(**model_args)

                # Log the response (stream events will be logged individually)
                log_openai_response("responses.create", result)
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to OpenAI: %s", err)
                raise HomeAssistantError("Error talking to OpenAI") from err

            async for content in chat_log.async_add_delta_content_stream(
                user_input.agent_id, _transform_stream(chat_log, result)
            ):
                # We no longer need the custom handler since tools are executed through their async_call method
                messages.extend(_convert_content_to_param(content))

            if not chat_log.unresponded_tool_results:
                break

        intent_response = intent.IntentResponse(language=user_input.language)
        assert type(chat_log.content[-1]) is conversation.AssistantContent
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)
