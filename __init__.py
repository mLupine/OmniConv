"""The OmniConv integration."""

from __future__ import annotations

import base64
from mimetypes import guess_type

import openai
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .api import async_setup_api
from .const import (
    CONF_BASE_URL,
    CONF_ORGANIZATION,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_SKIP_AUTHENTICATION,
    LOGGER,
)
from .helpers import is_azure
from .services import async_setup_services

PLATFORMS = (Platform.CONVERSATION,)

type OmniConvConfigEntry = ConfigEntry[openai.AsyncClient]


def encode_file(file_path: str) -> tuple[str, str]:
    """Return base64 version of file contents."""
    mime_type, _ = guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as image_file:
        return (mime_type, base64.b64encode(image_file.read()).decode("utf-8"))


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OmniConv."""
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: OmniConvConfigEntry) -> bool:
    """Set up OmniConv from a config entry."""
    base_url = entry.options.get(CONF_BASE_URL, DEFAULT_CONF_BASE_URL)
    skip_authentication = entry.options.get(
        CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
    )
    organization = entry.options.get(CONF_ORGANIZATION)

    _ = await async_setup_api(hass, entry)

    # Configure client based on whether it's Azure OpenAI or standard OpenAI
    if is_azure(base_url):
        # For Azure OpenAI
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=entry.options.get(CONF_API_KEY, "-"),
            azure_endpoint=base_url,
            api_version=entry.options.get("api_version", "2023-12-01-preview"),
            organization=organization,
            http_client=get_async_client(hass),
        )
    else:
        # For standard OpenAI
        client = openai.AsyncOpenAI(
            api_key=entry.options.get(CONF_API_KEY, "-"),
            base_url=base_url,
            organization=organization,
            http_client=get_async_client(hass),
        )

    # Cache current platform data which gets added to each request (caching done by library)
    _ = await hass.async_add_executor_job(client.platform_headers)

    # Skip authentication if configured to do so (useful for running against local inference servers)
    if not skip_authentication:
        try:
            if is_azure(base_url):
                # Azure API doesn't have models.list, so we just verify we can connect
                await client.with_options(timeout=10.0).deployments.list()
            else:
                await hass.async_add_executor_job(
                    client.with_options(timeout=10.0).models.list
                )
        except openai.AuthenticationError as err:
            LOGGER.error("Invalid API key: %s", err)
            return False
        except openai.OpenAIError as err:
            raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
