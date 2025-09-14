import pytest_asyncio
from homeassistant.core import HomeAssistant
from homeassistant import config_entries


@pytest_asyncio.fixture
async def hass(tmp_path):
    hass = HomeAssistant(str(tmp_path))
    hass.config_entries = config_entries.ConfigEntries(hass, {})
    await hass.async_start()
    yield hass
    await hass.async_stop()
