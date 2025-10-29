"""The OmniConv integration."""

from __future__ import annotations

from types import MappingProxyType

import openai
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_BASE_URL,
    CONF_ORGANIZATION,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_NAME,
    DEFAULT_SKIP_AUTHENTICATION,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
)
from .helpers import is_azure
from .services import async_setup_services

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)

type OmniConvConfigEntry = ConfigEntry[openai.AsyncClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OmniConv."""
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: OmniConvConfigEntry) -> bool:
    """Set up OmniConv from a config entry."""
    base_url = entry.data.get(CONF_BASE_URL, DEFAULT_CONF_BASE_URL)
    skip_authentication = entry.data.get(CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION)
    organization = entry.data.get(CONF_ORGANIZATION)
    api_key = entry.data.get(CONF_API_KEY, "-")

    if is_azure(base_url):
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=entry.data.get("api_version", "2023-12-01-preview"),
            organization=organization,
            http_client=get_async_client(hass),
        )
    else:
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            http_client=get_async_client(hass),
        )

    _ = await hass.async_add_executor_job(client.platform_headers)

    if not skip_authentication:
        try:
            if is_azure(base_url):
                await client.with_options(timeout=10.0).deployments.list()
            else:
                await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
        except openai.AuthenticationError as err:
            LOGGER.error("Invalid API key: %s", err)
            return False
        except openai.OpenAIError as err:
            raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OmniConv."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: OmniConvConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: OmniConvConfigEntry) -> bool:
    """Migrate entry."""
    LOGGER.debug("Migrating from version %s:%s", entry.version, entry.minor_version)

    if entry.version > 2:
        return False

    if entry.version == 1:
        new_data = entry.data.copy()
        conversation_subentry = ConfigSubentry(
            data=MappingProxyType(entry.options),
            subentry_type="conversation",
            title=entry.title or DEFAULT_CONVERSATION_NAME,
            unique_id=None,
        )
        ai_task_subentry = ConfigSubentry(
            data=MappingProxyType(RECOMMENDED_AI_TASK_OPTIONS),
            subentry_type="ai_task_data",
            title=DEFAULT_AI_TASK_NAME,
            unique_id=None,
        )

        hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            options={},
            title=DEFAULT_NAME,
            version=2,
            minor_version=1,
        )
        hass.config_entries.async_add_subentry(entry, conversation_subentry)
        hass.config_entries.async_add_subentry(entry, ai_task_subentry)

    LOGGER.debug("Migration to version %s:%s successful", entry.version, entry.minor_version)

    return True
