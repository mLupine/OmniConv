"""Config flow for Extended OpenAI Conversation integration."""

from __future__ import annotations

import json
import logging
import types
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol
import yaml
from homeassistant import config_entries
from homeassistant.components.zone import ENTITY_ID_HOME
from homeassistant.const import (
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_NAME,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from openai._exceptions import APIConnectionError, AuthenticationError
from voluptuous_openapi import convert

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_ENTITIES_PROMPT,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITY,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    CONF_WEB_SEARCH_COUNTRY,
    CONF_WEB_SEARCH_REGION,
    CONF_WEB_SEARCH_TIMEZONE,
    CONF_WEB_SEARCH_USER_LOCATION,
    CONTEXT_TRUNCATE_STRATEGIES,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_ENTITIES_PROMPT,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_WEB_SEARCH,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
    RECOMMENDED_WEB_SEARCH_USER_LOCATION,
    UNSUPPORTED_MODELS,
)
from .helpers import is_azure, validate_authentication

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): str,
    }
)

DEFAULT_CONF_FUNCTIONS_STR = yaml.dump(DEFAULT_CONF_FUNCTIONS, sort_keys=False)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_ENTITIES_PROMPT: DEFAULT_ENTITIES_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: (
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION
        ),
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_FUNCTIONS: DEFAULT_CONF_FUNCTIONS_STR,
        CONF_ATTACH_USERNAME: DEFAULT_ATTACH_USERNAME,
        CONF_USE_TOOLS: DEFAULT_USE_TOOLS,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    }
)

DEFAULT_INTEGRATION_OPTIONS = {
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_ENTITIES_PROMPT: DEFAULT_ENTITIES_PROMPT,
}


async def validate_options(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the options input allows us to connect.

    Data has the API keys and connection details from the options form.
    """
    skip_authentication = data.get(
        CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
    )

    # Only validate if authentication is not skipped and API key is provided
    if not skip_authentication and CONF_API_KEY in data:
        api_key = data[CONF_API_KEY]
        base_url = data.get(CONF_BASE_URL)
        api_version = data.get(CONF_API_VERSION)
        organization = data.get(CONF_ORGANIZATION)

        if base_url == DEFAULT_CONF_BASE_URL:
            # Do not set base_url if using OpenAI for case of OpenAI's base_url change
            base_url = None
            data.pop(CONF_BASE_URL, None)

        await validate_authentication(
            hass=hass,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            organization=organization,
            skip_authentication=skip_authentication,
        )


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Extended OpenAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        # Create default options with API configuration
        options = dict(DEFAULT_INTEGRATION_OPTIONS)

        # Initialize this entry with just the name in data
        data = {CONF_NAME: user_input.get(CONF_NAME, DEFAULT_NAME)}

        return self.async_create_entry(
            title=data[CONF_NAME],
            data=data,
            options=options,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


class OptionsFlow(config_entries.OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options
        errors: dict[str, str] = {}

        if user_input is not None:
            # Check if CONF_SKIP_AUTHENTICATION has changed, which requires re-rendering
            current_skip_auth = options.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            )
            new_skip_auth = user_input.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            )
            rerender_form = current_skip_auth != new_skip_auth

            if not rerender_form:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)

                if user_input.get(CONF_CHAT_MODEL) in UNSUPPORTED_MODELS:
                    errors[CONF_CHAT_MODEL] = "model_not_supported"

                # Validate API configuration if provided
                try:
                    if CONF_API_KEY in user_input:
                        await validate_options(self.hass, user_input)
                except APIConnectionError:
                    errors["base"] = "cannot_connect"
                except AuthenticationError:
                    errors["base"] = "invalid_auth"
                except Exception:  # pylint: disable=broad-except
                    _LOGGER.exception("Unexpected exception")
                    errors["base"] = "unknown"

                if user_input.get(CONF_WEB_SEARCH) and user_input.get(
                    CONF_WEB_SEARCH_USER_LOCATION
                ):
                    user_input.update(await self.get_location_data())

                if not errors:
                    # If functions are provided as a string, validate they're a proper YAML structure
                    if CONF_FUNCTIONS in user_input and isinstance(
                        user_input[CONF_FUNCTIONS], str
                    ):
                        try:
                            yaml.safe_load(user_input[CONF_FUNCTIONS])
                        except yaml.YAMLError as err:
                            errors[CONF_FUNCTIONS] = f"Invalid YAML: {err}"

                if not errors:
                    return self.async_create_entry(title="", data=user_input)
            else:
                # Re-render the options with the updated settings
                # Create a new options dictionary for rendering
                options = {
                    CONF_PROMPT: user_input[CONF_PROMPT],
                    CONF_ENTITIES_PROMPT: user_input[CONF_ENTITIES_PROMPT],
                    CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
                }

                # Include the skip_authentication value if it exists
                if CONF_SKIP_AUTHENTICATION in user_input:
                    options[CONF_SKIP_AUTHENTICATION] = user_input[
                        CONF_SKIP_AUTHENTICATION
                    ]

        schema = openai_config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    async def get_location_data(self) -> dict[str, str]:
        """Get approximate location data of the user."""
        location_data: dict[str, str] = {}
        zone_home = self.hass.states.get(ENTITY_ID_HOME)
        if zone_home is not None:
            # Configure client based on whether it's Azure OpenAI or standard OpenAI
            base_url = self.config_entry.options.get(
                CONF_BASE_URL, DEFAULT_CONF_BASE_URL
            )
            if is_azure(base_url):
                # Skip location detection for Azure OpenAI
                _LOGGER.debug("Skipping location detection with Azure OpenAI")
            else:
                client = openai.AsyncOpenAI(
                    api_key=self.config_entry.options[CONF_API_KEY],
                    base_url=base_url,
                    organization=self.config_entry.options.get(CONF_ORGANIZATION),
                    http_client=get_async_client(self.hass),
                )
                location_schema = vol.Schema(
                    {
                        vol.Optional(
                            CONF_WEB_SEARCH_CITY,
                            description="Free text input for the city, e.g. `San Francisco`",
                        ): str,
                        vol.Optional(
                            CONF_WEB_SEARCH_REGION,
                            description="Free text input for the region, e.g. `California`",
                        ): str,
                    }
                )
                response = await client.responses.create(
                    model=RECOMMENDED_CHAT_MODEL,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "Where are the following coordinates located: "
                                f"({zone_home.attributes[ATTR_LATITUDE]},"
                                f" {zone_home.attributes[ATTR_LONGITUDE]})?"
                            ),
                        }
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "approximate_location",
                            "description": (
                                "Approximate location data of the user "
                                "for refined web search results"
                            ),
                            "schema": convert(location_schema),
                            "strict": False,
                        }
                    },
                    store=False,
                )
                location_data = location_schema(json.loads(response.output_text) or {})

        if self.hass.config.country:
            location_data[CONF_WEB_SEARCH_COUNTRY] = self.hass.config.country
        location_data[CONF_WEB_SEARCH_TIMEZONE] = self.hass.config.time_zone

        _LOGGER.debug("Location data: %s", location_data)

        return location_data


def openai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> dict:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    skip_auth = options.get(CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION)

    schema: dict = {
        vol.Optional(
            CONF_SKIP_AUTHENTICATION, default=DEFAULT_SKIP_AUTHENTICATION
        ): bool,
    }

    # API key is only required if skip_authentication is False
    if not skip_auth:
        schema[vol.Required(CONF_API_KEY)] = str
    else:
        schema[vol.Optional(CONF_API_KEY)] = str

    # Add the rest of the authentication fields
    schema.update(
        {
            vol.Optional(CONF_BASE_URL, default=DEFAULT_CONF_BASE_URL): str,
            vol.Optional(CONF_API_VERSION): str,
            vol.Optional(CONF_ORGANIZATION): str,
            vol.Optional(
                CONF_PROMPT,
                description={
                    "suggested_value": options.get(
                        CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                    )
                },
            ): TemplateSelector(),
            vol.Optional(
                CONF_ENTITIES_PROMPT,
                description={
                    "suggested_value": options.get(
                        CONF_ENTITIES_PROMPT, DEFAULT_ENTITIES_PROMPT
                    )
                },
            ): TemplateSelector(),
            vol.Optional(
                CONF_LLM_HASS_API,
                description={"suggested_value": options.get(CONF_LLM_HASS_API)},
                default="none",
            ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        }
    )

    # Add all the model settings and extended options
    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_REASONING_EFFORT,
                description={"suggested_value": options.get(CONF_REASONING_EFFORT)},
                default=RECOMMENDED_REASONING_EFFORT,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["low", "medium", "high"],
                    translation_key=CONF_REASONING_EFFORT,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_WEB_SEARCH,
                description={"suggested_value": options.get(CONF_WEB_SEARCH)},
                default=RECOMMENDED_WEB_SEARCH,
            ): bool,
            vol.Optional(
                CONF_WEB_SEARCH_CONTEXT_SIZE,
                description={
                    "suggested_value": options.get(CONF_WEB_SEARCH_CONTEXT_SIZE)
                },
                default=RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["low", "medium", "high"],
                    translation_key=CONF_WEB_SEARCH_CONTEXT_SIZE,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_WEB_SEARCH_USER_LOCATION,
                description={
                    "suggested_value": options.get(CONF_WEB_SEARCH_USER_LOCATION)
                },
                default=RECOMMENDED_WEB_SEARCH_USER_LOCATION,
            ): bool,
            # Extended component settings
            vol.Optional(
                CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                description={
                    "suggested_value": options.get(
                        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION
                    )
                },
                default=DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            ): int,
            vol.Optional(
                CONF_FUNCTIONS,
                description={"suggested_value": options.get(CONF_FUNCTIONS)},
                default=DEFAULT_CONF_FUNCTIONS_STR,
            ): TemplateSelector(),
            vol.Optional(
                CONF_ATTACH_USERNAME,
                description={"suggested_value": options.get(CONF_ATTACH_USERNAME)},
                default=DEFAULT_ATTACH_USERNAME,
            ): BooleanSelector(),
            vol.Optional(
                CONF_USE_TOOLS,
                description={"suggested_value": options.get(CONF_USE_TOOLS)},
                default=DEFAULT_USE_TOOLS,
            ): BooleanSelector(),
            vol.Optional(
                CONF_CONTEXT_THRESHOLD,
                description={"suggested_value": options.get(CONF_CONTEXT_THRESHOLD)},
                default=DEFAULT_CONTEXT_THRESHOLD,
            ): int,
            vol.Optional(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                description={
                    "suggested_value": options.get(CONF_CONTEXT_TRUNCATE_STRATEGY)
                },
                default=DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=strategy["key"], label=strategy["label"])
                        for strategy in CONTEXT_TRUNCATE_STRATEGIES
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }
    )
    return schema
