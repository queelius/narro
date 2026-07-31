import pytest

from muse.modalities.image_vectorization.protocol import (
    VectorizationOutputError,
)
from muse.modalities.image_vectorization.svg import (
    MAX_DEPTH,
    MAX_ELEMENTS,
    validate_static_svg,
)


def test_extracts_svg_from_markdown_and_derives_geometry():
    raw = """Here is the result:
```svg
<svg viewBox="0 0 120 80"><rect width="120" height="80" fill="#fff"/></svg>
```
"""
    result = validate_static_svg(raw)
    assert result.svg.startswith("<svg")
    assert 'xmlns="http://www.w3.org/2000/svg"' in result.svg
    assert result.width == 120
    assert result.height == 80
    assert result.view_box == (0, 0, 120, 80)


def test_accepts_empty_svg_document():
    result = validate_static_svg("model output: <svg/> trailing text")
    assert result.svg.startswith("<svg")
    assert 'xmlns="http://www.w3.org/2000/svg"' in result.svg


def test_preserves_standard_gradients_and_local_references():
    raw = """
<svg xmlns="http://www.w3.org/2000/svg" width="40px" height="20px">
  <defs>
    <linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient>
  </defs>
  <path d="M0 0 L40 20" fill="url(#g)" style="stroke:#000;stroke-width:2"/>
</svg>
"""
    result = validate_static_svg(raw)
    assert "linearGradient" in result.svg
    assert "url(#g)" in result.svg
    assert result.width == 40
    assert result.height == 20


@pytest.mark.parametrize("payload", [
    "<svg><script>alert(1)</script></svg>",
    '<svg><foreignObject><p>html</p></foreignObject></svg>',
    '<svg><image href="data:image/png;base64,AAAA"/></svg>',
    '<svg><use href="https://example.com/x.svg#p"/></svg>',
    '<svg><path onclick="alert(1)" d="M0 0"/></svg>',
    '<svg><path fill="url(https://example.com/x)"/></svg>',
    '<svg><path style="fill:url(data:image/png;base64,AAAA)"/></svg>',
    '<svg><style>@import "https://example.com/x.css";</style></svg>',
    '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg/>',
    '<?xml-stylesheet href="https://example.com/x.css"?><svg/>',
])
def test_rejects_active_or_external_content(payload):
    with pytest.raises(VectorizationOutputError):
        validate_static_svg(payload)


def test_rejects_missing_or_incomplete_svg():
    with pytest.raises(VectorizationOutputError, match="no <svg>"):
        validate_static_svg("not a drawing")
    with pytest.raises(VectorizationOutputError, match="incomplete"):
        validate_static_svg("<svg><path/>")


def test_rejects_unknown_elements():
    with pytest.raises(VectorizationOutputError, match="not allowed"):
        validate_static_svg("<svg><filter/></svg>")


def test_rejects_elements_from_foreign_namespaces():
    with pytest.raises(VectorizationOutputError, match="namespace"):
        validate_static_svg(
            '<svg xmlns:x="https://example.com/x"><x:path d="M0 0"/></svg>'
        )


def test_rejects_attributes_from_foreign_namespaces():
    with pytest.raises(VectorizationOutputError, match="namespace"):
        validate_static_svg(
            '<svg xmlns:x="https://example.com/x">'
            '<path x:payload="value" d="M0 0"/></svg>'
        )


def test_rejects_too_many_elements():
    raw = "<svg>" + "<path/>" * MAX_ELEMENTS + "</svg>"
    with pytest.raises(VectorizationOutputError, match="elements"):
        validate_static_svg(raw)


def test_rejects_excessive_depth():
    raw = "<svg>" + "<g>" * MAX_DEPTH + "</g>" * MAX_DEPTH + "</svg>"
    with pytest.raises(VectorizationOutputError, match="nesting"):
        validate_static_svg(raw)


def test_percentage_dimensions_remain_unknown_without_viewbox():
    result = validate_static_svg(
        '<svg width="100%" height="100%"><circle cx="5" cy="5" r="5"/></svg>'
    )
    assert result.width is None
    assert result.height is None


def test_nonfinite_geometry_is_not_exposed_in_response_metadata():
    result = validate_static_svg(
        '<svg width="1e999" viewBox="nan 0 10 10"><path d="M0 0"/></svg>'
    )
    assert result.width is None
    assert result.height is None
    assert result.view_box is None
