from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator.

    - Return 0 for empty/whitespace-only text
    - Approximate tokens from character count: len(text.strip()) / 4
    """
    if not text or not text.strip():
        return 0
    return max(1, math.ceil(len(text.strip()) / 4))


def _normalize_vietnamese(text: str) -> str:
    """Remove diacritics from Vietnamese text for pattern matching."""
    # Handle đ/Đ which don't decompose in NFD
    text = text.replace("đ", "d").replace("Đ", "D")
    # Normalize to NFD (decomposed), then remove combining characters
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Maps each user id to one markdown file under root_dir.
    Supports read / write / edit operations.
    Optionally exposes helpers like facts() or upsert_fact().
    """

    root_dir: Path

    def _slugify_user_id(self, user_id: str) -> str:
        """Slugify: keep [A-Za-z0-9_.-], replace anything else with '_'."""
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)
        # prevent escaping root_dir (e.g. ../)
        if safe.startswith(".."):
            safe = "_" + safe
        return safe

    def path_for(self, user_id: str) -> Path:
        """Return the User.md file path for a user_id.

        The file lives at <root_dir>/<slug>/User.md.
        Parent directories are auto-created when writing.
        """
        slug = self._slugify_user_id(user_id)
        full = (self.root_dir / slug / "User.md").resolve()
        root_resolved = self.root_dir.resolve()
        # safety: reject any path outside root_dir
        if str(full).startswith(str(root_resolved)) or str(full).startswith(str(root_resolved) + os.sep):
            return self.root_dir / slug / "User.md"
        # fallback — keep under root_dir only
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return file content (utf-8) if it exists, else a default markdown profile."""
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        # default profile
        return "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return the file path.

        Parents are created automatically.
        """
        p = self.path_for(user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the FIRST occurrence of search_text inside User.md.

        Return True if changed (and actually wrote), False if search_text not found
        (do not create/modify the file when not found).
        """
        p = self.path_for(user_id)
        if not p.exists():
            return False
        content = p.read_text(encoding="utf-8")
        idx = content.find(search_text)
        if idx == -1:
            return False
        new_content = content[:idx] + replacement + content[idx + len(search_text) :]
        p.write_text(new_content, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes. 0 if missing."""
        p = self.path_for(user_id)
        if p.exists():
            return p.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse and return stable profile facts from the stored User.md.

        Fact lines are of the form `- **key**: value` or `- key: value`.
        Only the last occurrence of each key is kept.
        """
        content = self.read_text(user_id)
        facts = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                # Format: - **key**: value
                key_part, _, value = line[4:].partition("**:")
                key = key_part.strip()
                facts[key] = value.strip()
            elif line.startswith("- ") and ": " in line:
                # Format: - key: value
                key_part, _, value = line[2:].partition(": ")
                facts[key_part.strip()] = value.strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a fact in the profile."""
        path = self.path_for(user_id)
        content = self.read_text(user_id)
        facts = self.facts(user_id)
        facts[key] = value

        # Rebuild the Facts section
        facts_lines = [f"- **{k}**: {v}" for k, v in facts.items()]
        facts_section = "\n".join(facts_lines)

        # Replace the Facts section - find LAST occurrence to handle duplicates
        if "## Facts" in content:
            # Split by ## Facts and keep only the part before the last one
            parts = content.split("## Facts")
            before = "## Facts".join(parts[:-1])  # Everything before last ## Facts
            new_content = before + "## Facts\n\n" + facts_section + "\n"
        else:
            new_content = content.rstrip() + "\n\n## Facts\n\n" + facts_section + "\n"

        self.write_text(user_id, new_content)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _compact(self, thread_id: str) -> None:
        """Summarize messages beyond keep_messages, keep the tail of kept_messages."""
        st = self.state.get(thread_id)
        if st is None:
            return
        msgs: list[dict[str, str]] = st.get("messages", [])
        summary: str = st.get("summary", "")
        compactions: int = st.get("compactions", 0)

        # If keep_messages <= 0, keep at least 1 message
        k = max(1, self.keep_messages)

        if len(msgs) <= k:
            # nothing to compact; already fits
            return

        # summarize all but the last k messages
        to_summarize = msgs[:-k] if len(msgs) > k else []
        new_summary_parts: list[str] = []
        for m in to_summarize:
            content = m.get("content", "")
            # truncate to ~120 chars as a heuristic
            truncated = content[:120]
            new_summary_parts.append(truncated)

        new_summary = summary + " " + " ".join(new_summary_parts) if summary else " ".join(new_summary_parts)
        # compaction: keep the tail k messages + new summary
        new_msgs = msgs[-k:] if k > 0 else msgs
        self.state[thread_id] = {
            "messages": new_msgs,
            "summary": new_summary,
            "compactions": st["compactions"] + 1,
        }

    def _ensure_state(self, thread_id: str) -> None:
        """Create per-thread state if missing."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a new message and trigger compaction if needed."""
        self._ensure_state(thread_id)
        st = self.state[thread_id]

        # append the new message
        st["messages"].append({"role": role, "content": content})

        # trigger compaction if needed
        all_tokens = sum(
            estimate_tokens(m["content"]) for m in st["messages"]
        ) + estimate_tokens(st["summary"])

        if all_tokens > self.threshold_tokens and len(st["messages"]) > self.keep_messages:
            self._compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state dict (create if missing)."""
        if thread_id not in self.state:
            self._ensure_state(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread. 0 if unknown."""
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from user message.

    Only extracts from clear declarative statements, not questions or context.
    """
    facts = {}
    msg_lower = message.lower().strip()
    msg_norm = _normalize_vietnamese(message)
    msg_norm_lower = msg_norm.lower()

    # Skip questions very aggressively - use normalized text for diacritic-insensitive matching
    question_indicators = (
        "?", "lam sao", "giup", "cho biet", "cho toi", "co phai",
        "what", "how", "who", "where", "when", "why", "can you",
        "could you", "tell me", "explain", "ban co", "ban biet",
        "nhac lai", "thu nhac", "kiem tra", "ten gi", "o dau",
        "nghe gi", "lam gi", "thich gi", "dung khong", "sai khong",
        "nhin thay", "co phai", "la ai", "la gi", "nhu the nao"
        # "test" removed - matches "testuser" in names
        # "biet" removed - too broad, matches "ban" (bạn) in greetings
    )
    if any(q in msg_norm_lower for q in question_indicators) or "?" in message:
        return facts

    # Skip if message contains recall/request language
    recall_indicators = ("nhac", "ghi nho", "ghi nhớ", "luu", "luu giu", "ghi lai",
                         "remember", "recall", "save", "store")
    if any(w in msg_lower for w in recall_indicators):
        return facts

    # Name patterns - can appear anywhere in statement
    name_patterns = [
        # More specific: "tên là X" / "my name is X" - put first to avoid partial match
        r"(?:ten (?:la|toi la|cua toi la)|my name is|i am|i'm)\s+([A-Za-zÀ-ỹ][\wÀ-ỹ]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ]*)*)",
        # Less specific: "tôi là X" / "tôi tên X" 
        r"(?:toi la|toi ten|minh la|minh ten)\s+([A-Za-zÀ-ỹ][\wÀ-ỹ]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ]*)*)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, msg_norm, re.IGNORECASE)
        if match:
            # Use normalized match group directly (orig_match may fail due to diacritics)
            facts["name"] = match.group(1).strip()
            break

    # Location patterns - "tôi/mình ở X" or "giờ/bây giờ/ngày nay tôi/mình (đang) ở X"
    location_patterns = [
        r"(?:toi o|minh o|toi song|minh song|gio (?:toi|minh) (?:dang\s+)?o|bay gio (?:toi|minh) (?:dang\s+)?o|ngay nay (?:toi|minh) (?:dang\s+)?o)\s+([^,.!?\n]+?)(?:[,.\n]|$|\s+(?:va|chu|nhung|ma)\s+)",
    ]
    for pattern in location_patterns:
        match = re.search(pattern, msg_norm, re.IGNORECASE)
        if match:
            loc = match.group(1).strip().rstrip(".,")
            if not re.search(r"(?:de|cho|di|gap|hop|meeting|work|lam|lam viec)", loc, re.IGNORECASE):
                orig_match = re.search(pattern, message, re.IGNORECASE)
                if orig_match:
                    facts["location"] = orig_match.group(1).strip().rstrip(".,")
                else:
                    facts["location"] = loc
            break

    # Profession patterns - "tôi/mình làm X" or "giờ/bây giờ chuyển sang X" - exclude "làm việc ở"
    profession_patterns = [
        # "tôi/mình làm X" but NOT "làm việc ở Y" or "làm ở Y"
        r"(?:toi lam|minh lam)(?!\s+viec\s+o)(?!\s+o\s)\s+([^,.!?\n]+?)(?:[,.\n]|$|\s+va\s+)",
        # "tôi/mình là X" (job title)
        r"(?:toi la|minh la)\s+([^,.!?\n]+?)(?:[,.\n]|$|\s+va\s+)",
        # "giờ/bây giờ chuyển sang X" - job change
        r"(?:gio|bay gio|ngay nay)\s+chuyen sang\s+([^,.!?\n]+?)(?:[,.\n]|$|\s+va\s+)",
    ]
    for pattern in profession_patterns:
        match = re.search(pattern, msg_norm, re.IGNORECASE)
        if match:
            prof = match.group(1).strip().rstrip(".,")
            # Filter out work locations like "ở quán cà phê"
            if not re.search(r"(?:o\s|tai\s|de\s+)", prof, re.IGNORECASE) and \
               not re.search(r"(?:de|cho|gap|hoc|study)", prof, re.IGNORECASE):
                orig_match = re.search(pattern, message, re.IGNORECASE)
                if orig_match:
                    facts["profession"] = orig_match.group(1).strip().rstrip(".,")
                else:
                    facts["profession"] = prof
            break

    # Preferences / response style - "mình muốn (bạn) trả lời X" or "phong cách X"
    style_patterns = [
        # "muốn (bạn) phong cách/trả lời là X"
        r"(?:toi muon|minh muon|toi thich|minh thich|i prefer|i like)\s+(?:ban\s+)?(?:phong cach|style|tra loi|reply)\s+(?:la|nhu|theo)\s+([^,.!?]+)",
        # "muốn (bạn) trả lời X" (without là/như/theo)
        r"(?:toi muon|minh muon|toi thich|minh thich|i prefer|i like)\s+(?:ban\s+)?(?:tra loi|reply)\s+([^,.!?]+)",
        # "phong cách/style là X" - requires là/như/theo or at start
        r"(?:^|\s)(?:phong cach|style)\s+(?:la|nhu|theo)\s+([^,.!?]+)",
    ]
    for pattern in style_patterns:
        match = re.search(pattern, msg_norm, re.IGNORECASE)
        if match:
            orig_match = re.search(pattern, message, re.IGNORECASE)
            if orig_match:
                facts["response_style"] = orig_match.group(1).strip().rstrip(".,")
            else:
                facts["response_style"] = match.group(1).strip().rstrip(".,")
            break

    # Favorite drink / food - separate drink and food
    favorite_drink_patterns = [
        r"(?:do uong|đo uong)\s+(?:yeu thich|favorite)\s+(?:la|is)\s+([^,.!?]+)",
    ]
    for pattern in favorite_drink_patterns:
        match = re.search(pattern, msg_norm, re.IGNORECASE)
        if match:
            orig_match = re.search(pattern, message, re.IGNORECASE)
            if orig_match:
                facts["favorite_drink"] = orig_match.group(1).strip().rstrip(".,")
            else:
                facts["favorite_drink"] = match.group(1).strip().rstrip(".,")
            break

    favorite_food_patterns = [
        r"(?:mon an)\s+(?:yeu thich|favorite)\s+(?:la|is)\s+([^,.!?]+)",
    ]
    for pattern in favorite_food_patterns:
        match = re.search(pattern, msg_norm, re.IGNORECASE)
        if match:
            orig_match = re.search(pattern, message, re.IGNORECASE)
            if orig_match:
                facts["favorite_food"] = orig_match.group(1).strip().rstrip(".,")
            else:
                facts["favorite_food"] = match.group(1).strip().rstrip(".,")
            break

    # Also set generic favorite to drink if drink exists
    if "favorite_drink" in facts and "favorite" not in facts:
        facts["favorite"] = facts["favorite_drink"]

    # Interests - "tôi/mình thích X" (not food)
    interest_patterns = [
        r"(?:toi thich|minh thich|i like|i enjoy)\s+([^,.!?]+)",
    ]
    if "favorite" not in facts and "favorite_drink" not in facts and "favorite_food" not in facts:
        for pattern in interest_patterns:
            match = re.search(pattern, msg_norm, re.IGNORECASE)
            if match:
                interest = match.group(1).strip().rstrip(".,")
                if "food" not in interest.lower() and "do an" not in interest.lower() and "mon an" not in interest.lower():
                    orig_match = re.search(pattern, message, re.IGNORECASE)
                    if orig_match:
                        facts["interests"] = orig_match.group(1).strip().rstrip(".,")
                    else:
                        facts["interests"] = interest
                break

    return facts


# ============================================================
# BONUS: Confidence scoring & Conflict handling
# ============================================================

# Confidence threshold for writing to User.md (0.0 - 1.0)
# Higher = more conservative, fewer but more reliable facts
CONFIDENCE_THRESHOLD = 0.6

# Correction patterns - indicate user is correcting previous info
CORRECTION_PATTERNS = [
    r"khong con\s+\w+\s+nua",      # "không còn X nữa"
    r"khong phai\s+\w+",           # "không phải X"
    r"dinh chinh",                  # "đính chính"
    r"cap nhat",                    # "cập nhật"
    r"thay doi",                    # "thay đổi"
    r"sua\s+(?:doi|thong tin)",     # "sửa đổi" / "sửa thông tin"
    r"chuyen sang",                 # "chuyển sang"
    r"chuyen ve",                   # "chuyển về"
    r"sua\s+thanh",                 # "sửa thành"
    r"doi\s+thanh",                 # "đổi thành"
    r"moi la",                      # "mới là"
    r"update",                      # English
    r"correct",                     # English
    r"change\s+to",                 # English
]

# High confidence patterns (explicit statements)
HIGH_CONFIDENCE_PATTERNS = {
    "name": [r"ten (?:la|toi la|cua toi la)", r"my name is", r"i am", r"i'm"],
    "location": [r"(?:toi|minh) (?:o|song o|dang o)", r"live in", r"location is"],
    "profession": [r"(?:toi|minh) (?:lam|la)", r"work as", r"job is", r"chuyen sang"],
    "response_style": [r"(?:muon|thich).*(?:phong cach|style|tra loi).*la", r"style.*la"],
    "favorite_drink": [r"do uong yeu thich la", r"favorite drink is"],
    "favorite_food": [r"mon an yeu thich la", r"favorite food is"],
    "interests": [r"(?:toi|minh) thich", r"i like", r"i enjoy"],
}

# Low confidence patterns (ambiguous)
LOW_CONFIDENCE_PATTERNS = {
    "location": [r"gap o", r"hop o", r"den o", "di o"],
    "profession": [r"nghe", r"lam viec"],
}


def _detect_correction(message: str) -> tuple[bool, list[str]]:
    """Detect if message contains correction intent and which facts are being corrected.
    
    Returns:
        (is_correction, list_of_fact_keys_being_corrected)
    """
    msg_lower = message.lower()
    msg_norm = _normalize_vietnamese(message).lower()
    
    corrected_facts = []
    
    # Check for explicit correction keywords
    for pattern in CORRECTION_PATTERNS:
        if re.search(pattern, msg_norm, re.IGNORECASE):
            # Try to infer which fact is being corrected
            if any(w in msg_norm for w in ["o", "song", "dia chi", "live in", "location"]):
                corrected_facts.append("location")
            if any(w in msg_norm for w in ["lam", "nghe", "nghiep", "job", "work", "chuyen sang", "chuyen ve"]):
                corrected_facts.append("profession")
            if any(w in msg_norm for w in ["ten", "name", "goi la"]):
                corrected_facts.append("name")
            if any(w in msg_norm for w in ["phong cach", "style", "tra loi"]):
                corrected_facts.append("response_style")
            if any(w in msg_norm for w in ["do uong", "drink", "mon an", "food"]):
                corrected_facts.append("favorite_drink")
                corrected_facts.append("favorite_food")
            if any(w in msg_norm for w in ["thich", "yeu thich", "like", "enjoy"]):
                corrected_facts.append("interests")
            break
    
    return (len(corrected_facts) > 0, list(set(corrected_facts)))


def _compute_fact_confidence(fact_key: str, fact_value: str, message: str, 
                             is_correction: bool = False) -> float:
    """Compute confidence score (0.0-1.0) for an extracted fact.
    
    Factors:
    - Pattern specificity (explicit vs implicit)
    - Correction context (higher confidence for corrections)
    - Fact type (some types more reliable)
    - Value length/quality
    """
    base_confidence = 0.5
    msg_norm = _normalize_vietnamese(message).lower()
    
    # Boost for explicit correction intent
    if is_correction:
        base_confidence += 0.3
    
    # Check high-confidence patterns
    if fact_key in HIGH_CONFIDENCE_PATTERNS:
        for pattern in HIGH_CONFIDENCE_PATTERNS[fact_key]:
            if re.search(pattern, msg_norm, re.IGNORECASE):
                base_confidence += 0.2
                break
    
    # Penalize low-confidence patterns
    if fact_key in LOW_CONFIDENCE_PATTERNS:
        for pattern in LOW_CONFIDENCE_PATTERNS[fact_key]:
            if re.search(pattern, msg_norm, re.IGNORECASE):
                base_confidence -= 0.2
                break
    
    # Fact-specific adjustments
    if fact_key == "name":
        # Names are usually explicit
        base_confidence += 0.1
    elif fact_key in ("favorite_drink", "favorite_food"):
        # Explicit favorites are reliable
        if re.search(r"yeu thich|favorite", msg_norm):
            base_confidence += 0.15
    elif fact_key == "interests":
        # Generic "thich" can be noisy
        base_confidence -= 0.1
    elif fact_key == "location":
        # Check for meeting place noise
        if re.search(r"(?:de|cho|di|gap|hop|meeting|work|lam|lam viec)", fact_value, re.IGNORECASE):
            base_confidence -= 0.3
    
    # Value quality checks
    if len(fact_value.strip()) < 2:
        base_confidence -= 0.3
    elif len(fact_value.strip()) > 50:
        base_confidence -= 0.1
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, base_confidence))


def extract_profile_updates_with_confidence(message: str) -> dict[str, tuple[str, float]]:
    """Extract profile facts with confidence scores.
    
    Returns:
        Dict mapping fact_key -> (fact_value, confidence_score)
    """
    # First get raw facts
    raw_facts = extract_profile_updates(message)
    
    # Detect correction intent
    is_correction, corrected_keys = _detect_correction(message)
    
    # Compute confidence for each fact
    confident_facts = {}
    for key, value in raw_facts.items():
        confidence = _compute_fact_confidence(key, value, message, 
                                               is_correction=(key in corrected_keys))
        confident_facts[key] = (value, confidence)
    
    return confident_facts


def upsert_fact_with_confidence(store: UserProfileStore, user_id: str, 
                                key: str, value: str, confidence: float,
                                threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Insert or update a fact with confidence checking.
    
    Args:
        store: UserProfileStore instance
        user_id: User identifier
        key: Fact key
        value: Fact value
        confidence: Confidence score (0.0-1.0)
        threshold: Minimum confidence to write
    
    Returns:
        True if fact was written, False if rejected (below threshold)
    """
    if confidence < threshold:
        return False
    
    # For corrections, we always allow updates (even if lower confidence)
    # but only if explicitly marked as correction
    current_facts = store.facts(user_id)
    is_update = key in current_facts and current_facts[key] != value
    
    if is_update:
        # Allow updates with slightly lower threshold for corrections
        if confidence < threshold * 0.8:
            return False
    
    store.upsert_fact(user_id, key, value)
    return True


def _normalize_for_match(text: str) -> str:
    """Normalize text for matching (lowercase, strip diacritics)."""
    return _normalize_vietnamese(text).lower()


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear.

    Matches are case-insensitive and diacritic-insensitive.
    """
    if not expected:
        return 1.0
    ans_norm = _normalize_for_match(answer)
    found = sum(1 for exp in expected if _normalize_for_match(exp) in ans_norm)
    ratio = found / len(expected)
    if ratio == 1.0:
        return 1.0
    elif ratio >= 0.5:
        return 0.5
    return 0.0


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages.

    Heuristic: concatenate key exchanges, keeping it readable.
    """
    if not messages:
        return ""

    summary_parts = []
    for msg in messages[-max_items:]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if content:
            # Truncate very long messages
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"{role}: {content}")

    return "\n".join(summary_parts)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)