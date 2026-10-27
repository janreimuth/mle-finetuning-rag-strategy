"""Shared data, provider, and retrieval helpers for the strategy notebooks."""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, OpenAIError
from pydantic import BaseModel

load_dotenv(dotenv_path=Path(".env"), override=False)

DATA_PATH = Path("data") / "strategy_cases.json"
with DATA_PATH.open(encoding="utf-8") as file:
    DATA = json.load(file)

DOCUMENTS: list[dict[str, Any]] = DATA["documents"]
REQUESTS: list[dict[str, Any]] = DATA["support_requests"]
STYLE_EXAMPLES: list[dict[str, str]] = DATA["style_examples"]

ReplySource = Literal["live", "simulated", "fallback"]


class ModelReply(BaseModel):
    """Keep generated text and its execution source together."""

    text: str
    source: ReplySource


class ProviderConfig(BaseModel):
    """Settings for one OpenAI-compatible chat endpoint."""

    provider: str
    model: str
    base_url: str
    api_key: str | None = None
    dry_run: bool = True


def provider_order(preferred_provider: str | None = None) -> list[str]:
    """Return the preferred provider followed by the supported fallback."""

    requested = (preferred_provider or os.getenv("LLM_PROVIDER") or "groq").lower()
    if requested == "groq":
        return ["groq", "mistral"]
    if requested == "mistral":
        return ["mistral", "groq"]
    return [requested]


def provider_config_from_env(provider_name: str) -> ProviderConfig | None:
    """Return live provider settings when the matching API key exists."""

    if provider_name == "groq" and os.getenv("GROQ_API_KEY"):
        return ProviderConfig(
            provider="groq",
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            dry_run=False,
        )
    if provider_name == "mistral" and os.getenv("MISTRAL_API_KEY"):
        return ProviderConfig(
            provider="mistral",
            model=os.getenv("MISTRAL_MODEL", "ministral-14b-latest"),
            base_url="https://api.mistral.ai/v1",
            api_key=os.getenv("MISTRAL_API_KEY"),
            dry_run=False,
        )
    return None


def load_provider_config(preferred_provider: str | None = None) -> ProviderConfig:
    """Load live provider settings or deterministic dry-mode defaults."""

    order = provider_order(preferred_provider)
    for provider_name in order:
        config = provider_config_from_env(provider_name)
        if config is not None:
            return config
    return ProviderConfig(
        provider=order[0],
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        base_url="dry-run",
    )


CONFIG = load_provider_config()


def call_model(
    messages: list[dict[str, str]],
    *,
    dry_response: str,
    temperature: float = 0.2,
    response_format: dict[str, str] | None = None,
) -> ModelReply:
    """Return text labeled as live, simulated, or fallback after a provider error."""

    if CONFIG.dry_run or CONFIG.api_key is None:
        return ModelReply(text=dry_response, source="simulated")

    kwargs: dict[str, Any] = {
        "model": CONFIG.model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        with OpenAI(
            api_key=CONFIG.api_key,
            base_url=CONFIG.base_url,
            timeout=30,
            max_retries=3,
        ) as client:
            completion = client.chat.completions.create(**kwargs)
        return ModelReply(
            text=completion.choices[0].message.content or "", source="live"
        )
    except OpenAIError as error:
        # Keep diagnostics useful without printing response bodies or credentials.
        diagnostic = type(error).__name__
        if isinstance(error, APIStatusError):
            diagnostic += (
                f" (HTTP {error.status_code}, code={error.code or 'unspecified'})"
            )
        print(f"Live provider call failed: {diagnostic}")
        print("Using the deterministic response for this lesson run.")
        return ModelReply(text=dry_response, source="fallback")


def style_messages(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert approved examples into alternating user and assistant messages."""

    messages: list[dict[str, str]] = []
    for example in examples:
        messages.extend(
            [
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]},
            ]
        )
    return messages


def compact_style_examples(examples: list[dict[str, str]]) -> str:
    """Create a readable style summary for prompts and reports."""

    return "\n".join(
        f"- intent={example['intent']} tone={example['tone']}: {example['assistant']}"
        for example in examples
    )


def tokenize(text: str) -> list[str]:
    """Return lowercase word-like tokens for the transparent retriever."""

    return re.findall(r"[a-z0-9]+", text.lower())


def term_counts(text: str) -> Counter[str]:
    """Represent text as a sparse term-frequency vector."""

    return Counter(tokenize(text))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    """Compute cosine similarity between sparse term-frequency vectors."""

    shared_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in shared_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


DOCUMENT_VECTORS = {
    document["doc_id"]: term_counts(f"{document['title']} {document['text']}")
    for document in DOCUMENTS
}


def retrieve(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Rank policy documents while preserving source IDs for citations."""

    query_vector = term_counts(query)
    ranked = sorted(
        (
            {
                **document,
                "score": cosine(query_vector, DOCUMENT_VECTORS[document["doc_id"]]),
            }
            for document in DOCUMENTS
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    return ranked[:top_k]


def format_context(rows: list[dict[str, Any]]) -> str:
    """Format retrieved documents as compact, citeable prompt context."""

    return "\n".join(f"[{row['doc_id']}] {row['text']}" for row in rows)
