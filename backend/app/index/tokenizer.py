"""Tokenizer for BM25 that keeps clause ids and dotted numbers as single tokens."""

import re

# Tried in order: clause ids ("tep-4.2.1"), dotted numbers ("4.2.1"), then plain words.
TOKEN_RE = re.compile(r"[a-z]+-\d+(?:\.\d+)*|\d+(?:\.\d+)+|\w+")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # "20,000" -> "20000"
    return TOKEN_RE.findall(text)
