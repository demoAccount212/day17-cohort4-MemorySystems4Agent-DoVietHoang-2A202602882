from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Within-session memory only, no persistent User.md, forgets across threads."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # Optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if self.langchain_agent and not self.force_offline:
            return self._reply_langchain(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Simple offline behavior: store message, generate deterministic reply, update token counts."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        state = self.sessions[thread_id]

        # Store user message
        state.messages.append({"role": "user", "content": message})

        # Generate deterministic reply
        reply_text = self._generate_offline_reply(message)

        # Store assistant reply
        state.messages.append({"role": "assistant", "content": reply_text})

        # Update token counts
        user_tokens = estimate_tokens(message)
        assistant_tokens = estimate_tokens(reply_text)
        state.token_usage += user_tokens + assistant_tokens

        # For baseline, prompt tokens = all messages in session (no compaction)
        state.prompt_tokens_processed += user_tokens + assistant_tokens

        return {
            "reply": reply_text,
            "tokens_used": user_tokens + assistant_tokens,
            "prompt_tokens": state.prompt_tokens_processed,
        }

    def _generate_offline_reply(self, message: str) -> str:
        """Generate a simple deterministic response."""
        msg_lower = message.lower().strip()

        # Handle common questions
        if "tên" in msg_lower and ("gì" in msg_lower or "là" in msg_lower or "?" in message):
            return "Xin chào! Tôi không có thông tin tên của bạn trong phiên này."

        if "nghề" in msg_lower or "làm gì" in msg_lower or "job" in msg_lower:
            return "Tôi không nhớ nghề nghiệp của bạn trong phiên hiện tại."

        if "ở đâu" in msg_lower or "nơi ở" in msg_lower or "location" in msg_lower:
            return "Tôi không có thông tin nơi ở của bạn trong phiên này."

        if "style" in msg_lower or "phong cách" in msg_lower or "trả lời" in msg_lower:
            return "Tôi chưa nhận được hướng dẫn về phong cách trả lời trong phiên này."

        if "tóm tắt" in msg_lower or "summary" in msg_lower or "nhắc lại" in msg_lower:
            return "Trong phiên này tôi chưa lưu được thông tin dài hạn."

        # Default response
        return f"Đã nhận: {message[:50]}..."

    def _reply_langchain(self, thread_id: str, message: str) -> dict[str, Any]:
        """LangChain/LangGraph path (placeholder for live mode)."""
        # This would use the langchain_agent if built
        return self._reply_offline(thread_id, message)

    def _maybe_build_langchain_agent(self):
        """Optionally wire create_agent + InMemorySaver here."""
        try:
            model = build_chat_model(self.config.model)
            # In a full implementation, we would create a LangGraph agent here
            # with InMemorySaver for short-term memory
            self.langchain_agent = model
        except Exception:
            # Silently fail - will use offline mode
            self.langchain_agent = None