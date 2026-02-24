"""Extract speakable prose from Hugo/markdown content.

Provides two functions:
- parse_frontmatter(): separate YAML frontmatter from body
- extract_prose(): convert markdown body to speakable plain text
"""

from __future__ import annotations

import re
from typing import Any

import yaml


def parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown.

    Returns (metadata, body). If no frontmatter is found,
    returns ({}, full_input).
    """
    pattern = re.compile(r"\A---\n(.*?\n)---\n?(.*)", re.DOTALL)
    match = pattern.match(markdown)
    if not match:
        return {}, markdown
    frontmatter_str, body = match.group(1), match.group(2)
    try:
        meta = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError:
        return {}, markdown
    if not isinstance(meta, dict):
        return {}, markdown
    return meta, body


def extract_prose(text: str) -> str:
    """Convert markdown body to speakable plain text.

    Processing order:
    1. Strip fenced code blocks (``` or ~~~)
    2. Strip indented code blocks (4 spaces/tab after blank line)
    3. Strip LaTeX math: block $$...$$, \\[...\\], inline \\(...\\)
    4. Strip images ![alt](url)
    5. Strip Hugo shortcodes (paired and self-closing)
    6. Strip HTML tags
    7. Convert markdown links [text](url) -> text
    8. Strip heading markers
    9. Strip horizontal rules (---, ***, ___)
    10. Strip blockquote markers >
    11. Strip list markers (-, *, 1.)
    12. Strip bold/italic markers
    13. Strip inline code backticks (preserve text)
    14. Collapse excessive whitespace
    """
    # 1. Strip fenced code blocks (``` or ~~~)
    text = re.sub(r"^(`{3,}|~{3,}).*?\n.*?^\1\s*$", "", text, flags=re.MULTILINE | re.DOTALL)

    # 2. Strip indented code blocks (4 spaces or tab after a blank line)
    text = re.sub(r"(?<=\n\n)((?:(?:    |\t).+\n?)+)", "", text)

    # 3. Strip LaTeX math
    # Block math: $$...$$
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    # Block math: \[...\]
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    # Inline math: \(...\)
    text = re.sub(r"\\\(.*?\\\)", "", text)

    # 4. Strip images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # 5. Strip Hugo shortcodes
    # Paired shortcodes: {{< name ... >}}...{{< /name >}} (with body content)
    text = re.sub(
        r"\{\{<\s*(\w+)[^>]*>\}\}.*?\{\{<\s*/\s*\1\s*>\}\}",
        "", text, flags=re.DOTALL,
    )
    # Self-closing: {{< name ... />}}
    text = re.sub(r"\{\{<[^>]*/>\}\}", "", text)
    # Any remaining single shortcode tags (non-paired)
    text = re.sub(r"\{\{<[^>]*>\}\}", "", text)

    # 6. Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 7. Convert markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # 8. Strip heading markers (## Foo -> Foo)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 9. Strip horizontal rules (---, ***, ___ on their own line)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 10. Strip blockquote markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # 11. Strip list markers
    # Unordered: - or *
    text = re.sub(r"^(\s*)[*\-]\s+", r"\1", text, flags=re.MULTILINE)
    # Ordered: 1. 2. etc.
    text = re.sub(r"^(\s*)\d+\.\s+", r"\1", text, flags=re.MULTILINE)

    # 12. Strip bold/italic markers
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)

    # 13. Strip inline code backticks (preserve text inside)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 14. Collapse excessive whitespace (3+ newlines -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Final cleanup: strip leading/trailing whitespace
    text = text.strip()

    return text
