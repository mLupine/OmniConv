# ruff: noqa: E501
"""Constants for the OmniConv integration."""

import logging

from homeassistant.helpers import llm

DOMAIN = "omniconv"
LOGGER: logging.Logger = logging.getLogger(__package__)

DEFAULT_CONVERSATION_NAME = "OmniConv Conversation"
DEFAULT_AI_TASK_NAME = "OmniConv AI Task"
DEFAULT_NAME = "OmniConv"

CONF_ORGANIZATION = "organization"
CONF_BASE_URL = "base_url"
DEFAULT_CONF_BASE_URL = "https://api.openai.com/v1"
CONF_API_VERSION = "api_version"
CONF_SKIP_AUTHENTICATION = "skip_authentication"
DEFAULT_SKIP_AUTHENTICATION = False

CONF_CHAT_MODEL = "chat_model"
CONF_IMAGE_MODEL = "image_model"
CONF_CODE_INTERPRETER = "code_interpreter"
CONF_FILENAMES = "filenames"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"
CONF_ENTITIES_PROMPT = "entities_prompt"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_RECOMMENDED = "recommended"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_VERBOSITY = "verbosity"
CONF_PAYLOAD_TEMPLATE = "payload_template"

CONF_WEB_SEARCH = "web_search"
CONF_WEB_SEARCH_USER_LOCATION = "user_location"
CONF_WEB_SEARCH_CONTEXT_SIZE = "search_context_size"
CONF_WEB_SEARCH_CITY = "city"
CONF_WEB_SEARCH_REGION = "region"
CONF_WEB_SEARCH_COUNTRY = "country"
CONF_WEB_SEARCH_TIMEZONE = "timezone"

CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION = "max_function_calls_per_conversation"
CONF_FUNCTIONS = "functions"
CONF_ATTACH_USERNAME = "attach_username"
CONF_CONTEXT_THRESHOLD = "context_threshold"
CONF_CONTEXT_TRUNCATE_STRATEGY = "context_truncate_strategy"
CONF_PERFORMANCE_TRACING = "performance_tracing"

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 150
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 1
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 1
DEFAULT_ATTACH_USERNAME = False
DEFAULT_CONTEXT_THRESHOLD = 13000
DEFAULT_WEB_SEARCH = False
DEFAULT_WEB_SEARCH_CONTEXT_SIZE = "medium"
DEFAULT_WEB_SEARCH_USER_LOCATION = False
DEFAULT_CODE_INTERPRETER = False
DEFAULT_VERBOSITY = "medium"

CONF_LLM_HASS_API = "llm_hass_api"

RECOMMENDED_CHAT_MODEL = "gpt-4o-mini"
RECOMMENDED_IMAGE_MODEL = "gpt-image-1"
RECOMMENDED_MAX_TOKENS = 3000
RECOMMENDED_REASONING_EFFORT = "low"
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0
RECOMMENDED_WEB_SEARCH = False
RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE = "medium"
RECOMMENDED_WEB_SEARCH_USER_LOCATION = False
RECOMMENDED_CODE_INTERPRETER = False
RECOMMENDED_VERBOSITY = "medium"

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}
RECOMMENDED_AI_TASK_OPTIONS = {
    CONF_RECOMMENDED: True,
}

CONTEXT_TRUNCATE_STRATEGIES = [{"key": "clear", "label": "Clear All Messages"}]
DEFAULT_CONTEXT_TRUNCATE_STRATEGY = CONTEXT_TRUNCATE_STRATEGIES[0]["key"]

EVENT_AUTOMATION_REGISTERED = "automation_registered_via_omniconv"
EVENT_CONVERSATION_FINISHED = "omniconv.conversation.finished"

SERVICE_QUERY_IMAGE = "query_image"

DEFAULT_PROMPT = """I want you to act as smart home manager of Home Assistant.
I will provide information of smart home along with a question, you will truthfully make correction or answer using information provided in one sentence in everyday language.

Current Time: {{now()}}
Current Area: {{area_name(current_device_id)}}

The current state of devices is provided below. Use execute_services function only for requested action, not for current states.
Do not execute services without user's confirmation.
Do not restate or appreciate what user says, rather make a quick inquiry.
"""

DEFAULT_ENTITIES_PROMPT = """Available Devices:
```csv
entity_id,name,area_name,state,state_options,aliases
{% for entity in exposed_entities -%}
{%   if states[entity.entity_id] -%}
{{      entity.entity_id }},{{ entity.name }},{{area_name(entity.entity_id)}},{{ entity.state }},{{ states[entity.entity_id].attributes.options }},{{entity.aliases | join('/')}}
{%   endif -%}
{% endfor -%}
```
"""

DEFAULT_CONF_FUNCTIONS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Use this function to execute service of devices in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "description": "The domain of the service",
                                },
                                "service": {
                                    "type": "string",
                                    "description": "The service to be called",
                                },
                                "service_data": {
                                    "type": "object",
                                    "description": "The service data object to indicate what to" " control.",
                                    "properties": {
                                        "entity_id": {
                                            "type": "string",
                                            "description": (
                                                "The entity_id retrieved from available"
                                                " devices. It must start with domain,"
                                                " followed by dot character."
                                            ),
                                        }
                                    },
                                    "required": ["entity_id"],
                                },
                            },
                            "required": ["domain", "service", "service_data"],
                        },
                    }
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    }
]

UNSUPPORTED_MODELS: list[str] = [
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
]

UNSUPPORTED_WEB_SEARCH_MODELS: list[str] = [
    "gpt-5-nano",
    "gpt-3.5",
    "gpt-4-turbo",
    "gpt-4.1-nano",
    "o1",
    "o3-mini",
]

UNSUPPORTED_IMAGE_MODELS: list[str] = [
    "gpt-5-mini",
    "o3-mini",
    "o4",
    "o1",
    "gpt-3.5",
    "gpt-4-turbo",
]
