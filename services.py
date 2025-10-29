"""Services for the OmniConv integration."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import openai
import voluptuous as vol
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import selector
from homeassistant.helpers.typing import ConfigType
from openai._exceptions import OpenAIError
from openai.types.images_response import ImagesResponse
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
)

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FILENAMES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    SERVICE_QUERY_IMAGE,
)
from .entity import async_prepare_files_for_prompt
from .helpers import is_azure

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_GENERATE_CONTENT = "generate_content"

QUERY_IMAGE_SCHEMA = vol.Schema(
    {
        vol.Required("config_entry"): selector.ConfigEntrySelector(
            {
                "integration": DOMAIN,
            }
        ),
        vol.Required("model", default="gpt-4-vision-preview"): cv.string,
        vol.Required("prompt"): cv.string,
        vol.Required("images"): vol.All(cv.ensure_list, [{"url": cv.string}]),
        vol.Optional("max_tokens", default=300): cv.positive_int,
    }
)

_LOGGER = logging.getLogger(__package__)


def encode_file(file_path: str) -> tuple[str, str]:
    """Return base64 version of file contents."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as file:
        return (mime_type, base64.b64encode(file.read()).decode("utf-8"))


def to_image_param(hass: HomeAssistant, image) -> dict:
    """Convert url to base64 encoded image if local."""
    url = image["url"]

    if urlparse(url).scheme in cv.EXTERNAL_URL_PROTOCOL_SCHEMA_LIST:
        return image

    if not hass.config.is_allowed_path(url):
        raise HomeAssistantError(
            f"Cannot read `{url}`, no access to path; "
            "`allowlist_external_dirs` may need to be adjusted in "
            "`configuration.yaml`"
        )
    if not Path(url).exists():
        raise HomeAssistantError(f"`{url}` does not exist")
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type is None or not mime_type.startswith("image"):
        raise HomeAssistantError(f"`{url}` is not an image")

    image["url"] = f"data:{mime_type};base64,{encode_image(url)}"
    return image


def encode_image(image_path):
    """Convert to base64 encoded image."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def async_setup_services(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up services for the OmniConv component."""

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with dall-e."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data

        try:
            if is_azure(entry.data.get(CONF_BASE_URL, "")):
                raise HomeAssistantError("DALL-E image generation not supported with Azure OpenAI")

            response: ImagesResponse = await client.images.generate(
                model="dall-e-3",
                prompt=call.data[CONF_PROMPT],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        if not response.data or not response.data[0].url:
            raise HomeAssistantError("No image returned")

        return response.data[0].model_dump(exclude={"b64_json"})

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to ChatGPT and return the response."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        conversation_subentry = None
        for subentry in entry.subentries.values():
            if subentry.subentry_type == "conversation":
                conversation_subentry = subentry
                break

        if conversation_subentry:
            model: str = conversation_subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            max_tokens: int = conversation_subentry.data.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
            top_p: float = conversation_subentry.data.get(CONF_TOP_P, RECOMMENDED_TOP_P)
            temperature: float = conversation_subentry.data.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
            reasoning_effort: str = conversation_subentry.data.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)
        else:
            model = RECOMMENDED_CHAT_MODEL
            max_tokens = RECOMMENDED_MAX_TOKENS
            top_p = RECOMMENDED_TOP_P
            temperature = RECOMMENDED_TEMPERATURE
            reasoning_effort = RECOMMENDED_REASONING_EFFORT

        client: openai.AsyncClient = entry.runtime_data

        content: ResponseInputMessageContentListParam = [
            ResponseInputTextParam(type="input_text", text=call.data[CONF_PROMPT])
        ]

        if filenames := call.data.get(CONF_FILENAMES):
            for filename in filenames:
                if not hass.config.is_allowed_path(filename):
                    raise HomeAssistantError(
                        f"Cannot read `{filename}`, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted in "
                        "`configuration.yaml`"
                    )

            content.extend(
                await async_prepare_files_for_prompt(hass, [(Path(filename), None) for filename in filenames])
            )

        messages: ResponseInputParam = [EasyInputMessageParam(type="message", role="user", content=content)]

        model_args = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "user": call.context.user_id,
            "store": False,
        }

        if model.startswith("o"):
            model_args["reasoning"] = {"effort": reasoning_effort}

        if is_azure(entry.data.get(CONF_BASE_URL, "")):
            deployment = model
            model_args.pop("model")
            model_args["deployment"] = deployment

        try:
            response: Response = await client.responses.create(**model_args)

        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err
        except FileNotFoundError as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err

        return {"text": response.output_text}

    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Query an image."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        try:
            model = call.data["model"]
            images = [{"type": "image_url", "image_url": to_image_param(hass, image)} for image in call.data["images"]]

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": call.data["prompt"]}, *images],
                }
            ]
            _LOGGER.info("Prompt for %s: %s", model, messages)

            client = entry.runtime_data

            request_params = {
                "model": model,
                "messages": messages,
                "max_tokens": call.data["max_tokens"],
            }
            if is_azure(entry.data.get(CONF_BASE_URL, "")):
                request_params = {
                    "deployment_id": model,
                    "messages": messages,
                    "max_tokens": call.data["max_tokens"],
                }

            response = await client.chat.completions.create(**request_params)

            response_dict = response.model_dump()
        except OpenAIError as err:
            raise HomeAssistantError(f"Error querying image: {err}") from err

        return response_dict

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(cv.ensure_list, [cv.string]),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(("1024x1024", "1024x1792", "1792x1024")),
                vol.Optional("quality", default="standard"): vol.In(("standard", "hd")),
                vol.Optional("style", default="vivid"): vol.In(("vivid", "natural")),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_QUERY_IMAGE,
        query_image,
        schema=QUERY_IMAGE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
