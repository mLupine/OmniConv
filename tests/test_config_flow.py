import pathlib
import sys
from unittest.mock import patch

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from homeassistant.data_entry_flow import FlowResultType

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
