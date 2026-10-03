from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Configuration for the Day 17 Memory Systems lab.

    Attributes:
        base_dir: Repository root directory.
        data_dir: Directory containing benchmark datasets.
        state_dir: Directory for persistent state (User.md profiles, etc.).
        compact_threshold_tokens: Token threshold triggering conversation compaction.
        compact_keep_messages: Number of recent messages to keep after compaction.
        model: ProviderConfig for the main agent model.
        judge_model: ProviderConfig for the judge/evaluation model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _get_provider_defaults(provider: str) -> tuple[str, str | None, str | None]:
    """Return (default_model, api_key_env, base_url_env) for a provider."""
    defaults = {
        "openai": ("gpt-4o-mini", "OPENAI_API_KEY", None),
        "custom": ("gpt-4o-mini", "CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
        "gemini": ("gemini-1.5-flash", "GEMINI_API_KEY", None),
        "anthropic": ("claude-3-haiku-20240307", "ANTHROPIC_API_KEY", None),
        "ollama": ("llama3.1", None, "OLLAMA_BASE_URL"),
        "openrouter": ("openai/gpt-4o-mini", "OPENROUTER_API_KEY", None),
    }
    return defaults.get(provider, ("gpt-4o-mini", None, None))


def _build_provider_config(
    provider_env: str,
    model_env: str,
    temperature_env: str,
    default_provider: str,
    default_temperature: float = 0.0,
) -> ProviderConfig:
    """Build a ProviderConfig from environment variables with fallbacks."""
    provider = os.getenv(provider_env, default_provider).strip().lower()
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if provider not in supported:
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported: {', '.join(sorted(supported))}"
        )

    default_model, api_key_env, base_url_env = _get_provider_defaults(provider)
    model_name = os.getenv(model_env, default_model)
    temperature = float(os.getenv(temperature_env, str(default_temperature)))

    api_key = os.getenv(api_key_env) if api_key_env else None
    base_url = os.getenv(base_url_env) if base_url_env else None

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a fully populated LabConfig.

    Reads configuration from environment variables (optionally via .env file).
    Works in offline mode without any API keys set.

    Environment variables:
        LLM_PROVIDER: Main model provider (default: openai).
        LLM_MODEL: Main model name (provider-specific default).
        LLM_TEMPERATURE: Main model temperature (default: 0.0).
        LLM_JUDGE_PROVIDER: Judge model provider (falls back to main).
        LLM_JUDGE_MODEL: Judge model name (falls back to main).
        LLM_JUDGE_TEMPERATURE: Judge model temperature (falls back to main).
        LLM_COMPACT_THRESHOLD_TOKENS: Compaction trigger threshold (default: 1500).
        LLM_COMPACT_KEEP_MESSAGES: Messages to keep after compaction (default: 6).
        Provider-specific API keys and base URLs (all optional).

    Returns:
        LabConfig with all paths and provider settings resolved.
    """
    # Optionally load .env (guarded so module works without python-dotenv installed)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = int(os.getenv("LLM_COMPACT_THRESHOLD_TOKENS", "1500"))
    compact_keep_messages = int(os.getenv("LLM_COMPACT_KEEP_MESSAGES", "6"))

    # Main model config
    model = _build_provider_config(
        provider_env="LLM_PROVIDER",
        model_env="LLM_MODEL",
        temperature_env="LLM_TEMPERATURE",
        default_provider="openai",
        default_temperature=0.0,
    )

    # Judge model config (falls back to main model values)
    judge_model = _build_provider_config(
        provider_env="LLM_JUDGE_PROVIDER",
        model_env="LLM_JUDGE_MODEL",
        temperature_env="LLM_JUDGE_TEMPERATURE",
        default_provider=model.provider,
        default_temperature=model.temperature,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
