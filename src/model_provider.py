from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`."""
    value = value.strip().lower()
    aliases = {
        "openai": "openai",
        "gpt": "openai",
        "custom": "custom",
        "openai-compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "anthorpic": "anthropic",  # typo fix
        "ollama": "ollama",
        "local": "ollama",
        "openrouter": "openrouter",
        "router": "openrouter",
    }
    return aliases.get(value, value)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    raise ValueError(f"Unsupported provider: {provider}")