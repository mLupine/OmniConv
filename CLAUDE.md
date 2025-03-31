# Extended OpenAI Conversation - Development Guide

## Commands
- Install: Copy `custom_components/extended_openai_conversation` to your Home Assistant config directory
- Validate: Use Home Assistant's check_config: `hass --script check_config`
- Lint: `flake8 custom_components/extended_openai_conversation`
- Type check: `mypy custom_components/extended_openai_conversation`

## Code Style
- Follow [Home Assistant code style](https://developers.home-assistant.io/docs/development_guidelines)
- Use 4 spaces for indentation
- Group imports: stdlib → third-party → Home Assistant
- Type annotations required (Python 3.9+ style)
- snake_case for variables/functions, PascalCase for classes
- UPPER_CASE for constants
- Use docstrings for classes and methods
- Error handling: use custom exceptions from exceptions.py
- Follow Home Assistant component architecture patterns

## Notes
- This is a Home Assistant custom component extending OpenAI Conversation integration
- Respect existing patterns when modifying files