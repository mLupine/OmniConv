"""Base entity for OmniConv."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator, Callable
from mimetypes import guess_file_type
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import orjson

import openai
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify
from openai._streaming import AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCodeInterpreterToolCall,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearch,
    ResponseFunctionWebSearchParam,
    ResponseIncompleteEvent,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ToolParam,
    WebSearchToolParam,
)
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ImageGenerationCall as ImageGenerationCallParam,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.tool_param import (
    CodeInterpreter,
    CodeInterpreterContainerCodeInterpreterToolAuto,
    ImageGeneration,
)
from openai.types.responses.web_search_tool_param import UserLocation
import voluptuous as vol
from voluptuous_openapi import convert

from .const import (
    CONF_CHAT_MODEL,
    CONF_CODE_INTERPRETER,
    CONF_IMAGE_MODEL,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VERBOSITY,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITY,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    CONF_WEB_SEARCH_COUNTRY,
    CONF_WEB_SEARCH_REGION,
    CONF_WEB_SEARCH_TIMEZONE,
    CONF_WEB_SEARCH_USER_LOCATION,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CODE_INTERPRETER,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VERBOSITY,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
)
from .helpers import is_azure

if TYPE_CHECKING:
    from . import OmniConvConfigEntry

MAX_TOOL_ITERATIONS = 10


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with OpenAI API."""
    if schema["type"] == "object":
        schema.setdefault("strict", True)
        schema.setdefault("additionalProperties", False)
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)

    elif schema["type"] == "array":
        if "items" not in schema:
            return

        _adjust_schema(schema["items"])


def _format_structured_output(schema: vol.Schema, llm_api: llm.APIInstance | None) -> dict[str, Any]:
    """Format the schema to be compatible with OpenAI API."""
    result: dict[str, Any] = convert(
        schema,
        custom_serializer=(llm_api.custom_serializer if llm_api else llm.selector_serializer),
    )

    _adjust_schema(result)

    return result


def _serialize_json(obj: Any) -> str:
    """Serialize object to JSON using orjson for better performance."""
    return orjson.dumps(obj).decode("utf-8")


# Cache for content conversion
_content_conversion_cache: dict[int, tuple[list, list[str], dict]] = {}


def _convert_single_content(
    content: conversation.Content,
    web_search_calls: dict[str, ResponseFunctionWebSearchParam],
    reasoning_summary: list[str],
) -> tuple[list, list[str], dict[str, ResponseFunctionWebSearchParam]]:
    """Convert a single content item to OpenAI format.

    Returns: (messages, reasoning_summary_updates, web_search_calls_updates)
    """
    # Create a cache key based on content identity
    cache_key = id(content)

    # Check cache
    if cache_key in _content_conversion_cache:
        return _content_conversion_cache[cache_key]

    messages = []
    reasoning_updates = []
    web_search_updates = {}

    if isinstance(content, conversation.ToolResultContent):
        if content.tool_name == "web_search_call" and content.tool_call_id in web_search_calls:
            web_search_call = web_search_calls[content.tool_call_id].copy()
            web_search_call["status"] = content.tool_result.get("status", "completed")
            messages.append(web_search_call)
            # Mark for removal
            web_search_updates[content.tool_call_id] = None
        else:
            messages.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=content.tool_call_id,
                    output=_serialize_json(content.tool_result),
                )
            )
    elif content.content:
        role: Literal["user", "assistant", "system", "developer"] = content.role
        if role == "system":
            role = "developer"
        messages.append(EasyInputMessageParam(type="message", role=role, content=content.content))

    if isinstance(content, conversation.AssistantContent):
        if content.tool_calls:
            for tool_call in content.tool_calls:
                if tool_call.external and tool_call.tool_name == "web_search_call" and "action" in tool_call.tool_args:
                    web_search_updates[tool_call.id] = ResponseFunctionWebSearchParam(
                        type="web_search_call",
                        id=tool_call.id,
                        action=tool_call.tool_args["action"],
                        status="completed",
                    )
                else:
                    messages.append(
                        ResponseFunctionToolCallParam(
                            type="function_call",
                            name=tool_call.tool_name,
                            arguments=_serialize_json(tool_call.tool_args),
                            call_id=tool_call.id,
                        )
                    )

        if content.thinking_content:
            reasoning_updates.append(content.thinking_content)

        if isinstance(content.native, ResponseReasoningItem):
            messages.append(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id=content.native.id,
                    summary=(
                        [
                            {
                                "type": "summary_text",
                                "text": summary,
                            }
                            for summary in reasoning_summary + reasoning_updates
                        ]
                        if content.thinking_content
                        else []
                    ),
                    encrypted_content=content.native.encrypted_content,
                )
            )
            reasoning_updates = []  # Clear after using
        elif isinstance(content.native, ImageGenerationCall):
            messages.append(cast(ImageGenerationCallParam, content.native.to_dict()))

    # Cache the result
    result = (messages, reasoning_updates, web_search_updates)
    _content_conversion_cache[cache_key] = result

    # Limit cache size
    if len(_content_conversion_cache) > 100:
        # Remove oldest entries
        keys_to_remove = list(_content_conversion_cache.keys())[:20]
        for key in keys_to_remove:
            del _content_conversion_cache[key]

    return result


def _format_tool(tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None) -> FunctionToolParam:
    """Format tool specification."""
    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        description=tool.description,
        strict=False,
    )


def _convert_content_to_param(
    chat_content: list[conversation.Content],
) -> ResponseInputParam:
    """Convert any native chat message for this agent to the native format."""
    messages: ResponseInputParam = []
    reasoning_summary: list[str] = []
    web_search_calls: dict[str, ResponseFunctionWebSearchParam] = {}

    for content in chat_content:
        # Use cached conversion
        content_messages, reasoning_updates, web_search_updates = _convert_single_content(
            content, web_search_calls, reasoning_summary
        )

        # Apply updates
        messages.extend(content_messages)

        # Update reasoning summary
        if reasoning_updates:
            if isinstance(content, conversation.AssistantContent) and isinstance(content.native, ResponseReasoningItem):
                # Clear after using in reasoning item
                reasoning_summary = []
            else:
                reasoning_summary.extend(reasoning_updates)

        # Update web search calls
        for call_id, call in web_search_updates.items():
            if call is None:
                # Remove the call
                web_search_calls.pop(call_id, None)
            else:
                # Add or update the call
                web_search_calls[call_id] = call

    return messages


async def _transform_stream(
    chat_log: conversation.ChatLog,
    stream: AsyncStream[ResponseStreamEvent],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    last_summary_index = None
    last_role: Literal["assistant", "tool_result"] | None = None

    async for event in stream:
        LOGGER.debug("Received event: %s", event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                yield {"role": "assistant"}
                last_role = "assistant"
                last_summary_index = None
                current_tool_call = event.item
            elif (
                isinstance(event.item, ResponseOutputMessage)
                or (isinstance(event.item, ResponseReasoningItem) and last_summary_index is not None)
                or last_role != "assistant"
            ):
                yield {"role": "assistant"}
                last_role = "assistant"
                last_summary_index = None
        elif isinstance(event, ResponseOutputItemDoneEvent):
            if isinstance(event.item, ResponseReasoningItem):
                yield {
                    "native": ResponseReasoningItem(
                        type="reasoning",
                        id=event.item.id,
                        summary=[],
                        encrypted_content=event.item.encrypted_content,
                    )
                }
                last_summary_index = len(event.item.summary) - 1
            elif isinstance(event.item, ResponseCodeInterpreterToolCall):
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=event.item.id,
                            tool_name="code_interpreter",
                            tool_args={
                                "code": event.item.code,
                                "container": event.item.container_id,
                            },
                            external=True,
                        )
                    ]
                }
                yield {
                    "role": "tool_result",
                    "tool_call_id": event.item.id,
                    "tool_name": "code_interpreter",
                    "tool_result": {
                        "output": (
                            [output.to_dict() for output in event.item.outputs]
                            if event.item.outputs is not None
                            else None
                        )
                    },
                }
                last_role = "tool_result"
            elif isinstance(event.item, ResponseFunctionWebSearch):
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=event.item.id,
                            tool_name="web_search_call",
                            tool_args={
                                "action": event.item.action.to_dict(),
                            },
                            external=True,
                        )
                    ]
                }
                yield {
                    "role": "tool_result",
                    "tool_call_id": event.item.id,
                    "tool_name": "web_search_call",
                    "tool_result": {"status": event.item.status},
                }
                last_role = "tool_result"
            elif isinstance(event.item, ImageGenerationCall):
                yield {"native": event.item}
                last_summary_index = -1
        elif isinstance(event, ResponseTextDeltaEvent):
            yield {"content": event.delta}
        elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            if last_summary_index is not None and event.summary_index != last_summary_index:
                yield {"role": "assistant"}
                last_role = "assistant"
            last_summary_index = event.summary_index
            yield {"thinking_content": event.delta}
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


class OmniConvBaseLLMEntity(Entity):
    """OmniConv base LLM entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OmniConvConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._is_azure = is_azure(entry.data.get("base_url", ""))  # Cache Azure detection
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenAI",
            model=subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._cached_formatted_tools: list[ToolParam] | None = None
        self._cached_tools_hash: int | None = None

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        force_image: bool = False,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data

        messages = _convert_content_to_param(chat_log.content)

        model_args = ResponseCreateParamsStreaming(
            model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            input=messages,
            max_output_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            user=chat_log.conversation_id,
            store=False,
            stream=True,
        )

        if model_args["model"].startswith(("o", "gpt-5")):
            model_args["reasoning"] = {
                "effort": (
                    options.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)
                    if not model_args["model"].startswith("gpt-5-pro")
                    else "high"
                ),
                "summary": "auto",
            }
            model_args["include"] = ["reasoning.encrypted_content"]

        if model_args["model"].startswith("gpt-5"):
            model_args["text"] = {"verbosity": options.get(CONF_VERBOSITY, RECOMMENDED_VERBOSITY)}

        tools: list[ToolParam] = []
        if chat_log.llm_api:
            # Create hash of tool signatures to detect changes
            current_tools_hash = hash(tuple((tool.name, type(tool).__name__) for tool in chat_log.llm_api.tools))

            # Check if tools have changed or not cached yet
            if self._cached_formatted_tools is None or self._cached_tools_hash != current_tools_hash:
                self._cached_formatted_tools = [
                    _format_tool(tool, chat_log.llm_api.custom_serializer) for tool in chat_log.llm_api.tools
                ]
                self._cached_tools_hash = current_tools_hash
                LOGGER.debug("Formatted and cached %s tools", len(self._cached_formatted_tools))

            tools = self._cached_formatted_tools  # No need to copy, tools aren't mutated

        if options.get(CONF_WEB_SEARCH):
            web_search = WebSearchToolParam(
                type="web_search",
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
            tools.append(web_search)

        if options.get(CONF_CODE_INTERPRETER, RECOMMENDED_CODE_INTERPRETER):
            tools.append(
                CodeInterpreter(
                    type="code_interpreter",
                    container=CodeInterpreterContainerCodeInterpreterToolAuto(type="auto"),
                )
            )
            model_args.setdefault("include", []).append("code_interpreter_call.outputs")

        if force_image:
            from openai.types.responses import ToolChoiceTypesParam

            image_model = options.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)
            image_tool = ImageGeneration(
                type="image_generation",
                model=image_model,
                output_format="png",
            )
            if image_model == "gpt-image-1":
                image_tool["input_fidelity"] = "high"
            tools.append(image_tool)
            model_args["tool_choice"] = ToolChoiceTypesParam(type="image_generation")
            model_args["store"] = True

        if tools:
            model_args["tools"] = tools

        last_content = chat_log.content[-1]

        if last_content.role == "user" and last_content.attachments:
            files = await async_prepare_files_for_prompt(
                self.hass,
                [(a.path, a.mime_type) for a in last_content.attachments],
            )
            last_message = messages[-1]
            assert (
                last_message["type"] == "message"
                and last_message["role"] == "user"
                and isinstance(last_message["content"], str)
            )
            last_message["content"] = [
                {"type": "input_text", "text": last_message["content"]},
                *files,
            ]

        if structure and structure_name:
            model_args["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": slugify(structure_name),
                    "schema": _format_structured_output(structure, chat_log.llm_api),
                },
            }

        client = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                if self._is_azure:
                    deployment = model_args["model"]
                    model_args_azure = model_args.copy()
                    model_args_azure.pop("model")
                    model_args_azure["deployment"] = deployment
                    stream = await client.responses.create(**model_args_azure)
                else:
                    stream = await client.responses.create(**model_args)

                messages.extend(
                    _convert_content_to_param(
                        [
                            content
                            async for content in chat_log.async_add_delta_content_stream(
                                self.entity_id, _transform_stream(chat_log, stream)
                            )
                        ]
                    )
                )
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                if isinstance(err, openai.APIError) and err.type == "insufficient_quota":
                    LOGGER.error("Insufficient funds for OpenAI: %s", err)
                    raise HomeAssistantError("Insufficient funds for OpenAI") from err
                if "Verify Organization" in str(err):
                    ir.async_create_issue(
                        self.hass,
                        DOMAIN,
                        "organization_verification_required",
                        is_fixable=False,
                        is_persistent=False,
                        learn_more_url="https://help.openai.com/en/articles/10910291-api-organization-verification",
                        severity=ir.IssueSeverity.WARNING,
                        translation_key="organization_verification_required",
                        translation_placeholders={
                            "platform_settings": "https://platform.openai.com/settings/organization/general"
                        },
                    )

                LOGGER.error("Error talking to OpenAI: %s", err)
                raise HomeAssistantError("Error talking to OpenAI") from err

            if not chat_log.unresponded_tool_results:
                break


async def async_prepare_files_for_prompt(
    hass: HomeAssistant, files: list[tuple[Path, str | None]]
) -> ResponseInputMessageContentListParam:
    """Append files to a prompt.

    Caller needs to ensure that the files are allowed.
    """

    def append_files_to_content() -> ResponseInputMessageContentListParam:
        content: ResponseInputMessageContentListParam = []

        for file_path, provided_mime_type in files:
            if not file_path.exists():
                raise HomeAssistantError(f"`{file_path}` does not exist")

            mime_type = provided_mime_type or guess_file_type(file_path)[0]

            if not mime_type or not mime_type.startswith(("image/", "application/pdf")):
                raise HomeAssistantError(
                    "Only images and PDF are supported by the OpenAI API," f"`{file_path}` is not an image file or PDF"
                )

            base64_file = base64.b64encode(file_path.read_bytes()).decode("utf-8")

            if mime_type.startswith("image/"):
                content.append(
                    ResponseInputImageParam(
                        type="input_image",
                        image_url=f"data:{mime_type};base64,{base64_file}",
                        detail="auto",
                    )
                )
            elif mime_type.startswith("application/pdf"):
                content.append(
                    ResponseInputFileParam(
                        type="input_file",
                        filename=str(file_path),
                        file_data=f"data:{mime_type};base64,{base64_file}",
                    )
                )

        return content

    return await hass.async_add_executor_job(append_files_to_content)
