"""Checks on the model reply before it reaches the user."""

import json
import re
from dataclasses import dataclass, field
from typing import cast

from app.schemas import Hit, Status

VALID_STATUS = {"answered", "refused", "conflict"}
NOT_COVERED = "The policy documents do not cover this question."

# Common phrasings of instructions hidden in documents. A simple filter: it catches the usual
# patterns, not every possible wording, so the prompt rule against obeying evidence stays too.
INJECTION_PATTERNS = re.compile(
    r"ignore (all |any )?(previous|prior|above|earlier) instructions"
    r"|disregard (all |any )?(previous|prior|above|earlier)"
    r"|system override"
    r"|(note|message|instructions?) (for|to) (the )?(ai|assistants?|automated assistants?|language models?)",
    re.IGNORECASE,
)


def screen_evidence(hits: list[Hit]) -> tuple[list[Hit], list[str]]:
    """Drop clauses that contain instruction-like text before they reach the model."""
    kept, notes = [], []
    for hit in hits:
        if INJECTION_PATTERNS.search(hit.chunk.text):
            notes.append(f"left out {hit.chunk.chunk_id}: it contains instruction-like text")
        else:
            kept.append(hit)
    return kept, notes


@dataclass
class CheckedAnswer:
    status: Status
    answer: str
    citations: list[str]
    problems: list[str] = field(default_factory=list)


def parse_reply(text: str) -> dict | None:
    """Read the JSON object in the reply, also when the model wrapped it in other text."""
    for candidate in (text, *re.findall(r"\{.*\}", text, re.DOTALL)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def check_reply(text: str, evidence_ids: set[str]) -> CheckedAnswer:
    """Validate status and citations. An answer without a valid citation is turned into a refusal."""
    data = parse_reply(text)
    if data is None:
        return CheckedAnswer("error", "The model reply could not be read.", [], ["reply is not JSON"])

    problems: list[str] = []
    status = data.get("status")
    if status not in VALID_STATUS:
        return CheckedAnswer("error", "The model reply could not be read.", [], [f"unknown status {status!r}"])

    cited = [c for c in data.get("citations") or [] if isinstance(c, str)]
    citations = [c for c in dict.fromkeys(cited) if c in evidence_ids]
    if len(citations) < len(set(cited)):
        problems.append(f"removed citations not in the evidence: {sorted(set(cited) - evidence_ids)}")

    answer = str(data.get("answer", "")).strip()
    if status == "refused":
        return CheckedAnswer("refused", answer or NOT_COVERED, [], problems)
    if not citations:
        problems.append(f"{status} reply had no valid citation")
        return CheckedAnswer("refused", NOT_COVERED, [], problems)
    return CheckedAnswer(cast(Status, status), answer, citations, problems)
