from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: Three memory layers - within-session, persistent User.md, compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # Optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if self.langchain_agent and not self.force_offline:
            return self._reply_langchain(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return estimated prompt tokens processed for this thread."""
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Return User.md file size in bytes."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path with persistent memory and compaction."""
        # 1. Extract stable profile facts from the incoming message
        facts = extract_profile_updates(message)
        for key, value in facts.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Append message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context tokens (User.md + summary + recent messages)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 4. Generate response using persisted memory
        reply_text = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", reply_text)

        # 6. Update token counters
        user_tokens = estimate_tokens(message)
        assistant_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + user_tokens + assistant_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "reply": reply_text,
            "tokens_used": user_tokens + assistant_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn: User.md + summary + recent messages."""
        total = 0

        # Include User.md
        profile_text = self.profile_store.read_text(user_id)
        total += estimate_tokens(profile_text)

        # Include compact memory context
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        if summary:
            total += estimate_tokens(summary)

        # Include recent kept messages
        messages = ctx.get("messages", [])
        for msg in messages:
            total += estimate_tokens(msg.get("content", ""))

        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""
        msg_lower = message.lower().strip()

        # Get profile facts
        facts = self.profile_store.facts(user_id)
        name = facts.get("name", "")
        location = facts.get("location", "")
        profession = facts.get("profession", "")
        response_style = facts.get("response_style", "")
        interests = facts.get("interests", "")
        favorite_drink = facts.get("favorite_drink", "")
        favorite_food = facts.get("favorite_food", "")

        # Get compact memory context
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        recent_messages = ctx.get("messages", [])

        # Build context for response
        context_parts = []
        if name:
            context_parts.append(f"Tên: {name}")
        if location:
            context_parts.append(f"Nơi ở: {location}")
        if profession:
            context_parts.append(f"Nghề nghiệp: {profession}")
        if response_style:
            context_parts.append(f"Style: {response_style}")
        if interests:
            context_parts.append(f"Sở thích: {interests}")
        if favorite_drink:
            context_parts.append(f"Đồ uống yêu thích: {favorite_drink}")
        if favorite_food:
            context_parts.append(f"Món ăn yêu thích: {favorite_food}")

        # Add compact memory summary if available
        if summary:
            context_parts.append(f"Tóm tắt: {summary[:300]}")

        context_str = "; ".join(context_parts) if context_parts else "Chưa có thông tin"

        # Handle specific questions - check for multiple fact types
        asked_facts = []
        if "tên" in msg_lower and ("gì" in msg_lower or "là" in msg_lower or "?" in message):
            asked_facts.append(("name", name, "Tên bạn là {0}."))
        if "nghề" in msg_lower or "làm gì" in msg_lower or "job" in msg_lower or "nghiệp" in msg_lower:
            asked_facts.append(("profession", profession, "Bạn đang làm {0}."))
        if ("ở đâu" in msg_lower or "nơi ở" in msg_lower or "location" in msg_lower) and "?" in message:
            asked_facts.append(("location", location, "Bạn đang ở {0}."))
        if "style" in msg_lower or "phong cách" in msg_lower or ("trả lời" in msg_lower and "?" in message):
            asked_facts.append(("style", response_style, "Phong cách bạn thích: {0}."))
        if "sở thích" in msg_lower or "thích gì" in msg_lower or "hobby" in msg_lower:
            asked_facts.append(("interests", interests, "Bạn thích {0}."))
        if ("đồ uống" in msg_lower or "do uong" in msg_lower or "drink" in msg_lower) and ("yêu thích" in msg_lower or "favorite" in msg_lower or "thích" in msg_lower):
            asked_facts.append(("drink", favorite_drink, "Đồ uống yêu thích của bạn là {0}."))
        if ("món ăn" in msg_lower or "mon an" in msg_lower or "food" in msg_lower) and ("yêu thích" in msg_lower or "favorite" in msg_lower or "thích" in msg_lower):
            asked_facts.append(("food", favorite_food, "Món ăn yêu thích của bạn là {0}."))

        if asked_facts:
            answers = []
            for key, value, template in asked_facts:
                if value:
                    answers.append(template.format(value))
                else:
                    answers.append(f"Tôi chưa biết {key} của bạn.")
            return " ".join(answers)

        if "tóm tắt" in msg_lower or "summary" in msg_lower or "nhắc lại" in msg_lower:
            if context_parts:
                return "Thông tin tôi nhớ về bạn: " + context_str
            return "Tôi chưa có thông tin dài hạn về bạn."

        # Default: acknowledge with context
        return f"Đã ghi nhận. {context_str}"

    def _reply_langchain(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """LangChain/LangGraph path (placeholder for live mode)."""
        return self._reply_offline(user_id, thread_id, message)

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with tools and compact middleware."""
        try:
            model = build_chat_model(self.config.model)
            # In a full implementation, we would create a LangGraph agent here
            # with InMemorySaver for short-term memory, tools for User.md read/write,
            # and summarization middleware for long threads
            self.langchain_agent = model
        except Exception:
            self.langchain_agent = None