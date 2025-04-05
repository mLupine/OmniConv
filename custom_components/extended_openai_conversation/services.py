"""Services for the Extended OpenAI Conversation integration."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import openai
from openai.types.images_response import ImagesResponse
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
)
import voluptuous as vol

from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType
from openai._exceptions import OpenAIError

from .const import (
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
from .helpers import is_azure, log_openai_request, log_openai_response

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
    """Set up services for the Extended OpenAI Conversation component."""

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

        client = entry.runtime_data

        try:
            if is_azure(entry.options.get("base_url", "")):
                # Azure OpenAI uses different API for DALL-E
                raise HomeAssistantError(
                    "DALL-E image generation not supported with Azure OpenAI"
                )

            # Create request parameters
            request_params = {
                "model": "dall-e-3",
                "prompt": call.data[CONF_PROMPT],
                "size": call.data["size"],
                "quality": call.data["quality"],
                "style": call.data["style"],
                "response_format": "url",
                "n": 1,
            }

            # Log the request parameters
            log_openai_request("images.generate", **request_params)

            # Make the API call
            response: ImagesResponse = await client.images.generate(**request_params)

            # Log the response
            log_openai_response("images.generate", response)
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

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

        model: str = entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        client = entry.runtime_data

        content: ResponseInputMessageContentListParam = [
            ResponseInputTextParam(type="input_text", text=call.data[CONF_PROMPT])
        ]

        def append_files_to_content() -> None:
            for filename in call.data[CONF_FILENAMES]:
                if not hass.config.is_allowed_path(filename):
                    raise HomeAssistantError(
                        f"Cannot read `{filename}`, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted in "
                        "`configuration.yaml`"
                    )
                if not Path(filename).exists():
                    raise HomeAssistantError(f"`{filename}` does not exist")
                mime_type, base64_file = encode_file(filename)
                if "image/" in mime_type:
                    content.append(
                        ResponseInputImageParam(
                            type="input_image",
                            file_id=filename,
                            image_url=f"data:{mime_type};base64,{base64_file}",
                            detail="auto",
                        )
                    )
                elif "application/pdf" in mime_type:
                    content.append(
                        ResponseInputFileParam(
                            type="input_file",
                            filename=filename,
                            file_data=f"data:{mime_type};base64,{base64_file}",
                        )
                    )
                else:
                    raise HomeAssistantError(
                        "Only images and PDF are supported by the OpenAI API,"
                        f"`{filename}` is not an image file or PDF"
                    )

        if CONF_FILENAMES in call.data:
            await hass.async_add_executor_job(append_files_to_content)

        messages: ResponseInputParam = [
            EasyInputMessageParam(type="message", role="user", content=content)
        ]

        try:
            model_args = {
                "model": model,
                "input": messages,
                "max_output_tokens": entry.options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                "top_p": entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": entry.options.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                "user": call.context.user_id,
                "store": False,
            }

            if model.startswith("o"):
                model_args["reasoning"] = {
                    "effort": entry.options.get(
                        CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                    )
                }

            # Handle Azure OpenAI if applicable
            if is_azure(entry.options.get("base_url", "")):
                # Azure OpenAI uses deployments instead of model names
                deployment = model
                # Azure uses 'deployment' parameter instead of 'model'
                model_args.pop("model")
                model_args["deployment"] = deployment

            # Log the request parameters
            log_openai_request("responses.create", **model_args)

            # Make the API call
            response: Response = await client.responses.create(**model_args)

            # Log the response
            log_openai_response("responses.create", response)

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
            images = [
                {"type": "image_url", "image_url": to_image_param(hass, image)}
                for image in call.data["images"]
            ]

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": call.data["prompt"]}] + images,
                }
            ]
            _LOGGER.info("Prompt for %s: %s", model, messages)

            client = entry.runtime_data

            # Prepare request parameters
            if is_azure(entry.options.get("base_url", "")):
                # For Azure OpenAI, we need to use the deployment name
                request_params = {
                    "deployment_id": model,  # Azure uses deployment_id instead of model
                    "messages": messages,
                    "max_tokens": call.data["max_tokens"],
                }
            else:
                request_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": call.data["max_tokens"],
                }

            # Log the request parameters
            log_openai_request("chat.completions.create", **request_params)

            # Make the API call
            response = await client.chat.completions.create(**request_params)

            # Log the response
            log_openai_response("chat.completions.create", response)

            response_dict = response.model_dump()
        except OpenAIError as err:
            raise HomeAssistantError(f"Error querying image: {err}") from err

        return response_dict

    # Register all services
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
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
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
                vol.Optional("size", default="1024x1024"): vol.In(
                    ("1024x1024", "1024x1792", "1792x1024")
                ),
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
