"""Text normalization for YouTube comments."""

from __future__ import annotations

import html
import re
from typing import Mapping

try:
    import emoji
except ImportError:  # pragma: no cover - optional dependency fallback
    emoji = None


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """Limit repeated characters, e.g. cooooool -> cool."""
    pattern = re.compile(r"(.)\1{" + str(max_repeats) + r",}", re.IGNORECASE)
    return pattern.sub(lambda match: match.group(1) * max_repeats, text)


def replace_social_slang(text: str, slang_map: Mapping[str, str]) -> str:
    """Replace common social-media slang using whole-word matching."""
    if not slang_map:
        return text

    def replace(match: re.Match[str]) -> str:
        return slang_map.get(match.group(0).lower(), match.group(0))

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(key) for key in slang_map.keys()) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(replace, text)


def demojize_text(text: str) -> str:
    """Convert emoji to text aliases when the emoji package is available."""
    if emoji is None:
        return text
    return emoji.demojize(text, language="en")


def clean_text(
    value: object,
    slang_map: Mapping[str, str] | None = None,
    max_repeated_chars: int = 2,
) -> str:
    """Clean one social-media text value."""
    if value is None:
        return ""

    text = str(value).lower()
    text = html.unescape(text)
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = demojize_text(text)
    text = replace_social_slang(text, slang_map or {})
    text = normalize_repeated_characters(text, max_repeats=max_repeated_chars)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ_:\s]", " ", text)
    text = text.replace("_", " ").replace(":", " ")
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
