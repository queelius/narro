"""Extraction and security validation for model-generated SVG.

SVG is an active document format: scripts, event handlers, remote
references, and foreign objects can execute or fetch content in a
browser. Vectorization output is therefore treated as untrusted even
though it came from a model. This module accepts a deliberately static
subset suited to icons, diagrams, and Manim's SVG importer.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from muse.modalities.image_vectorization.protocol import (
    VectorizationOutputError,
)


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
XML_NS = "http://www.w3.org/XML/1998/namespace"
MAX_SVG_BYTES = 2 * 1024 * 1024
MAX_ELEMENTS = 10_000
MAX_DEPTH = 128
MAX_ATTRIBUTE_CHARS = 1_000_000

_SVG_START_RE = re.compile(r"<svg(?=[\s>/])", re.IGNORECASE)
_SVG_END_RE = re.compile(r"</svg\s*>", re.IGNORECASE)
_SVG_EMPTY_RE = re.compile(r"<svg(?=[\s>/])[^>]*?/\s*>", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"^[ \t]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:px|pt|pc|mm|cm|in)?[ \t]*$"
)
_URL_RE = re.compile(r"url\(\s*([^)]+?)\s*\)", re.IGNORECASE)

_ALLOWED_ELEMENTS = frozenset({
    "svg", "g", "defs", "title", "desc", "symbol", "use",
    "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
    "text", "tspan", "textPath",
    "linearGradient", "radialGradient", "stop", "pattern",
    "clipPath", "mask", "marker",
})
_ALLOWED_STYLE_PROPERTIES = frozenset({
    "fill", "fill-opacity", "fill-rule",
    "stroke", "stroke-opacity", "stroke-width", "stroke-linecap",
    "stroke-linejoin", "stroke-miterlimit", "stroke-dasharray",
    "stroke-dashoffset", "opacity", "color",
    "font-family", "font-size", "font-style", "font-weight",
    "text-anchor", "dominant-baseline", "letter-spacing",
    "clip-path", "mask", "marker-start", "marker-mid", "marker-end",
    "stop-color", "stop-opacity", "visibility", "display",
})
_URI_ATTRIBUTES = frozenset({
    "href", "fill", "stroke", "clip-path", "mask",
    "marker-start", "marker-mid", "marker-end",
})
_DANGEROUS_VALUE_FRAGMENTS = (
    "javascript:", "vbscript:", "data:", "file:", "http:", "https:",
    "@import", "expression(", "-moz-binding",
)


@dataclass(frozen=True)
class SvgInfo:
    """Validated SVG plus geometry useful to downstream renderers."""

    svg: str
    width: float | None
    height: float | None
    view_box: tuple[float, float, float, float] | None


def validate_static_svg(raw: str) -> SvgInfo:
    """Extract and validate one static SVG from generated model text.

    The model may surround its answer with Markdown or prose. Only the
    first complete ``<svg>...</svg>`` document is considered. Unsafe
    constructs are rejected rather than silently removed so callers
    never receive a materially altered drawing.
    """
    if not isinstance(raw, str):
        raise VectorizationOutputError("generated output is not text")
    lowered_raw = raw.lower()
    if "<!doctype" in lowered_raw or "<!entity" in lowered_raw:
        raise VectorizationOutputError("DTD and entity declarations are forbidden")
    if "<?xml-stylesheet" in lowered_raw:
        raise VectorizationOutputError("XML stylesheets are forbidden")

    start_match = _SVG_START_RE.search(raw)
    if start_match is None:
        raise VectorizationOutputError("generated output contains no <svg> root")
    end_match = _SVG_END_RE.search(raw, start_match.start())
    if end_match is None:
        empty_match = _SVG_EMPTY_RE.match(raw, start_match.start())
        if empty_match is None:
            raise VectorizationOutputError("generated SVG is incomplete")
        candidate = empty_match.group(0).strip()
    else:
        candidate = raw[start_match.start():end_match.end()].strip()
    if len(candidate.encode("utf-8")) > MAX_SVG_BYTES:
        raise VectorizationOutputError(
            f"generated SVG exceeds {MAX_SVG_BYTES} bytes"
        )

    try:
        root = ET.fromstring(candidate)
    except ET.ParseError as exc:
        raise VectorizationOutputError("generated SVG is not well-formed XML") from exc
    if _local_name(root.tag) != "svg":
        raise VectorizationOutputError("generated document root is not <svg>")

    count = 0
    attribute_chars = 0
    stack: list[tuple[ET.Element, int]] = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        count += 1
        if count > MAX_ELEMENTS:
            raise VectorizationOutputError(
                f"generated SVG exceeds {MAX_ELEMENTS} elements"
            )
        if depth > MAX_DEPTH:
            raise VectorizationOutputError(
                f"generated SVG exceeds nesting depth {MAX_DEPTH}"
            )

        name = _local_name(element.tag)
        namespace = _namespace(element.tag)
        if namespace not in ("", SVG_NS):
            raise VectorizationOutputError(
                f"SVG element <{name}> uses an unsupported namespace"
            )
        if name not in _ALLOWED_ELEMENTS:
            raise VectorizationOutputError(
                f"SVG element <{name}> is not allowed"
            )
        for raw_name, value in element.attrib.items():
            attr_name = _local_name(raw_name)
            attr_namespace = _namespace(raw_name)
            if attr_namespace not in ("", XLINK_NS, XML_NS):
                raise VectorizationOutputError(
                    f"SVG attribute {attr_name!r} uses an unsupported namespace"
                )
            if attr_namespace == XML_NS and attr_name != "space":
                raise VectorizationOutputError(
                    f"SVG xml:{attr_name} attribute is not allowed"
                )
            attribute_chars += len(raw_name) + len(value)
            if attribute_chars > MAX_ATTRIBUTE_CHARS:
                raise VectorizationOutputError(
                    "generated SVG attributes are too large"
                )
            if attr_name.lower().startswith("on"):
                raise VectorizationOutputError(
                    "SVG event-handler attributes are forbidden"
                )
            _validate_attribute(attr_name, value)
        stack.extend((child, depth + 1) for child in element)

    width = _dimension(root.get("width"))
    height = _dimension(root.get("height"))
    view_box = _view_box(root.get("viewBox") or root.get("viewbox"))
    if (width is None or height is None) and view_box is not None:
        width = width if width is not None else view_box[2]
        height = height if height is not None else view_box[3]

    # Use a stable default namespace rather than ElementTree's ns0 prefix.
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    serialized = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    if root.tag == "svg" and "xmlns" not in root.attrib:
        # Namespace-free model output remains namespace-free after parsing.
        # Add the standard namespace for portability across SVG consumers.
        root.set("xmlns", SVG_NS)
        serialized = ET.tostring(
            root, encoding="unicode", short_empty_elements=True,
        )
    return SvgInfo(
        svg=serialized,
        width=width,
        height=height,
        view_box=view_box,
    )


def _local_name(name: str) -> str:
    if "}" in name:
        return name.rsplit("}", 1)[-1]
    if ":" in name:
        return name.rsplit(":", 1)[-1]
    return name


def _namespace(name: str) -> str:
    if name.startswith("{") and "}" in name:
        return name[1:].split("}", 1)[0]
    return ""


def _validate_attribute(name: str, value: str) -> None:
    lowered = value.strip().lower()
    if any(fragment in lowered for fragment in _DANGEROUS_VALUE_FRAGMENTS):
        raise VectorizationOutputError(
            f"unsafe value in SVG attribute {name!r}"
        )

    if name == "style":
        _validate_style(value)
    if name == "href":
        if not value.strip().startswith("#"):
            raise VectorizationOutputError(
                "SVG href references must be local fragments"
            )
    if name in _URI_ATTRIBUTES or "url(" in lowered:
        for match in _URL_RE.finditer(value):
            target = match.group(1).strip().strip("'\"")
            if not target.startswith("#"):
                raise VectorizationOutputError(
                    "SVG URL references must be local fragments"
                )
        remainder = _URL_RE.sub("", value)
        if "url(" in remainder.lower():
            raise VectorizationOutputError("malformed SVG URL reference")


def _validate_style(value: str) -> None:
    for declaration in value.split(";"):
        declaration = declaration.strip()
        if not declaration:
            continue
        if ":" not in declaration:
            raise VectorizationOutputError("malformed inline SVG style")
        prop, prop_value = declaration.split(":", 1)
        if prop.strip().lower() not in _ALLOWED_STYLE_PROPERTIES:
            raise VectorizationOutputError(
                f"SVG style property {prop.strip()!r} is not allowed"
            )
        # Apply URL/scheme checks without recursively treating this as
        # another style attribute.
        lowered = prop_value.strip().lower()
        if any(fragment in lowered for fragment in _DANGEROUS_VALUE_FRAGMENTS):
            raise VectorizationOutputError("unsafe inline SVG style value")
        for match in _URL_RE.finditer(prop_value):
            target = match.group(1).strip().strip("'\"")
            if not target.startswith("#"):
                raise VectorizationOutputError(
                    "SVG style URL references must be local fragments"
                )


def _dimension(value: str | None) -> float | None:
    if value is None:
        return None
    match = _NUMBER_RE.match(value)
    if match is None:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) and number >= 0 else None


def _view_box(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    pieces = re.split(r"[\s,]+", value.strip())
    if len(pieces) != 4:
        return None
    try:
        box = tuple(float(piece) for piece in pieces)
    except ValueError:
        return None
    if not all(math.isfinite(piece) for piece in box):
        return None
    if box[2] < 0 or box[3] < 0:
        return None
    return box  # type: ignore[return-value]
