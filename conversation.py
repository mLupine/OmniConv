"""Conversation support for OpenAI."""

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers import llm, template
from homeassistant.helpers.llm import _get_exposed_entities

from . import OmniConvConfigEntry
from .const import CONF_ATTACH_ENTITIES, CONF_PROMPT, DOMAIN
from .entity import OmniConvBaseLLMEntity

# Max number of back and forth with the LLM to generate a response


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OmniConvConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [OmniConvConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class OmniConvConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    OmniConvBaseLLMEntity,
):
    """OmniConv conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, entry: OmniConvConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the API."""
        options = self.subentry.data
        attach = options.get(CONF_ATTACH_ENTITIES, True)
        prompt = options.get(CONF_PROMPT)
        llm_api = options.get(CONF_LLM_HASS_API) if attach else None

        if not attach:
            exposed = _get_exposed_entities(self.hass, DOMAIN)
            prompt = template.Template(prompt or llm.DEFAULT_INSTRUCTIONS_PROMPT, self.hass).async_render(
                {"exposed_entities": exposed}
            )

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_api,
                prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
