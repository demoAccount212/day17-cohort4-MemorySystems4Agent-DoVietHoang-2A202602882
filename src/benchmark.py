from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import load_conversations, recall_points


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.

    Scores based on:
    - Non-empty response
    - Contains expected facts (bonus)
    - Reasonable length (not too short, not too long)
    - Vietnamese language indicators
    """
    score = 0.0

    if not answer or not answer.strip():
        return 0.0

    # Base score for having a response
    score += 0.3

    # Bonus for containing expected facts
    if expected:
        found = sum(1 for exp in expected if exp.lower() in answer.lower())
        score += 0.4 * (found / len(expected))

    # Length check (reasonable response)
    length = len(answer.strip())
    if 10 <= length <= 500:
        score += 0.2
    elif length > 500:
        score += 0.1

    # Vietnamese language indicators
    vn_indicators = ["bạn", "tôi", "mình", "là", "của", "nhớ", "thông tin"]
    if any(ind in answer.lower() for ind in vn_indicators):
        score += 0.1

    return min(1.0, score)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores = []
    quality_scores = []
    user_id = conversations[0]["user_id"] if conversations else "unknown"

    # Feed all turns to the agent
    for conv in conversations:
        thread_id = conv["id"]
        for turn in conv["turns"]:
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get("tokens_used", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

    # Ask recall questions in a fresh thread
    recall_thread = f"{user_id}_recall"
    for conv in conversations:
        for rq in conv.get("recall_questions", []):
            result = agent.reply(user_id, recall_thread, rq["question"])
            answer = result.get("reply", "")
            recall_scores.append(recall_points(answer, rq.get("expected_contains", [])))
            quality_scores.append(heuristic_quality(answer, rq.get("expected_contains", [])))

    # Record memory file growth and compaction count
    memory_growth = 0
    compactions = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = agent.memory_file_size(user_id)
    if hasattr(agent, "compaction_count"):
        # Get max compactions across all threads
        for conv in conversations:
            compactions = max(compactions, agent.compaction_count(conv["id"]))
        compactions = max(compactions, agent.compaction_count(recall_thread))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a tabulated output."""
    try:
        from tabulate import tabulate

        headers = [
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ]
        table = []
        for row in rows:
            table.append([
                row.agent_name,
                row.agent_tokens_only,
                row.prompt_tokens_processed,
                f"{row.recall_score:.2f}",
                f"{row.response_quality:.2f}",
                row.memory_growth_bytes,
                row.compactions,
            ])
        return tabulate(table, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback to simple format
        lines = []
        headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
        lines.append(" | ".join(headers))
        lines.append(" | ".join(["---"] * len(headers)))
        for row in rows:
            lines.append(" | ".join([
                row.agent_name,
                str(row.agent_tokens_only),
                str(row.prompt_tokens_processed),
                f"{row.recall_score:.2f}",
                f"{row.response_quality:.2f}",
                str(row.memory_growth_bytes),
                str(row.compactions),
            ]))
        return "\n".join(lines)


def main() -> None:
    """Run both benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)

    # Load datasets
    data_dir = config.data_dir
    standard_convs = load_conversations(data_dir / "conversations.json")
    stress_convs = load_conversations(data_dir / "advanced_long_context.json")

    # Initialize agents in offline mode (deterministic, no API keys needed)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    print("=" * 80)
    print("STANDARD BENCHMARK (data/conversations.json)")
    print("=" * 80)

    baseline_row = run_agent_benchmark("Baseline", baseline, standard_convs, config)
    advanced_row = run_agent_benchmark("Advanced", advanced, standard_convs, config)

    print(format_rows([baseline_row, advanced_row]))
    print()

    print("=" * 80)
    print("LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
    print("=" * 80)

    # New agents for stress test to avoid state contamination
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    baseline_stress_row = run_agent_benchmark("Baseline", baseline_stress, stress_convs, config)
    advanced_stress_row = run_agent_benchmark("Advanced", advanced_stress, stress_convs, config)

    print(format_rows([baseline_stress_row, advanced_stress_row]))


if __name__ == "__main__":
    main()