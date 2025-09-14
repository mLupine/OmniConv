from unittest.mock import patch

import pytest
import yaml
import voluptuous as vol
from homeassistant.data_entry_flow import FlowResultType
from OmniConv import tools
from OmniConv.const import (
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_ORGANIZATION,
    CONF_SKIP_AUTH,
)
from OmniConv.config_flow import OmniConvConfigFlow


@pytest.mark.asyncio
async def test_user_flow_custom_fields(hass):
    flow = OmniConvConfigFlow()
    flow.hass = hass
    with patch("OmniConv.config_flow.validate_input"):
        result = await flow.async_step_user(
            {
                "api_key": "k",
                CONF_BASE_URL: "http://example",
                CONF_API_VERSION: "v1",
                CONF_ORGANIZATION: "org",
                CONF_SKIP_AUTH: True,
            }
        )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    data = result["data"]
    assert data[CONF_BASE_URL] == "http://example"
    assert data[CONF_SKIP_AUTH] is True


@pytest.mark.asyncio
async def test_user_flow_defaults(hass):
    flow = OmniConvConfigFlow()
    flow.hass = hass
    with patch("OmniConv.config_flow.validate_input"):
        result = await flow.async_step_user({"api_key": "k"})
    assert result["type"] == FlowResultType.CREATE_ENTRY
    data = result["data"]
    assert data.get(CONF_BASE_URL) is None
    assert CONF_SKIP_AUTH not in data


@pytest.mark.asyncio
async def test_validate_function_schema():
    cfg = "- type: template\n  name: t\n  description: d\n  template: 'hi'\n"
    parsed = yaml.safe_load(cfg)
    out = tools.validate_configs(parsed)
    assert out[0]["name"] == "t"


@pytest.mark.asyncio
async def test_validate_invalid_schema():
    with pytest.raises(vol.Invalid):
        tools.validate_configs([{"type": "template"}])
