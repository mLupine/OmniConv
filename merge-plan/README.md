# Merge Plan for Extended OpenAI Conversation

## Overview
The upstream OpenAI Conversation component has undergone a significant refactor. Key changes include:

1. Architecture shift from Chat Completions API to Response API
2. New services: `generate_image` and `generate_content`
3. New file: `conversation.py` for conversation handling
4. Enhanced media support (images, PDFs)
5. Web search capabilities for gpt-4o models
6. Improved integration with Home Assistant's LLM API

## Files to Update

### 1. manifest.json
- Update OpenAI requirement to 1.68.2
- Add `after_dependencies` for assist_pipeline & intent
- Update dependencies list

### 2. const.py
- Add new constants for web search & reasoning
- Update default models
- Add unsupported models list

### 3. config_flow.py
- Integrate recommended settings toggle
- Add web search configuration
- Add location detection for better search results
- Implement model validation

### 4. __init__.py
- Implement new services from upstream
- Use Response API instead of Chat Completions
- Integrate media handling capabilities

### 5. New Files
- Add `conversation.py` to handle conversation with new APIs
- Update `services.yaml` with new services

### 6. services.py
- Modify to use the same APIs as upstream
- Merge functionality with upstream image services

## Extended Features to Preserve
- Azure OpenAI support
- Custom services not in upstream (like image query)
- Extended configuration options

## Implementation Strategy
1. Start with upstream files as the base
2. Add extended functionality where needed
3. Ensure compatibility with existing configurations
4. Test with both standard OpenAI and Azure OpenAI