"""Tests for narro.hugo.extract — markdown to speakable plain text."""

from narro.hugo.extract import parse_frontmatter, extract_prose


class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_basic_frontmatter(self):
        md = '---\ntitle: "Hello"\ntts: true\n---\nBody text'
        meta, body = parse_frontmatter(md)
        assert meta["title"] == "Hello"
        assert meta["tts"] is True
        assert body.strip() == "Body text"

    def test_no_frontmatter(self):
        md = "Just plain text, no frontmatter here."
        meta, body = parse_frontmatter(md)
        assert meta == {}
        assert body == md

    def test_tts_false(self):
        md = "---\ntts: false\n---\nContent"
        meta, body = parse_frontmatter(md)
        assert meta["tts"] is False

    def test_no_tts_field(self):
        md = '---\ntitle: "Post"\nauthor: "Someone"\n---\nContent'
        meta, body = parse_frontmatter(md)
        assert "tts" not in meta
        assert meta["title"] == "Post"
        assert body.strip() == "Content"

    def test_invalid_yaml(self):
        """Invalid YAML in frontmatter returns empty meta and full input."""
        md = "---\n: :\n  bad: [yaml\n---\nBody"
        meta, body = parse_frontmatter(md)
        assert meta == {}
        assert body == md

    def test_non_dict_frontmatter(self):
        """Frontmatter that parses to a non-dict (e.g. a list) returns empty meta."""
        md = "---\n- item1\n- item2\n---\nBody"
        meta, body = parse_frontmatter(md)
        assert meta == {}
        assert body == md


class TestExtractProse:
    """Tests for markdown-to-speakable-text extraction."""

    def test_plain_text(self):
        text = "Hello, this is plain text."
        assert extract_prose(text) == "Hello, this is plain text."

    def test_strips_headings(self):
        text = "## Intro\n\nText"
        result = extract_prose(text)
        assert result == "Intro\n\nText"

    def test_strips_fenced_code(self):
        text = "Before code.\n\n```python\nprint('hello')\n```\n\nAfter code."
        result = extract_prose(text)
        assert "print" not in result
        assert "Before code." in result
        assert "After code." in result

    def test_strips_fenced_code_tilde(self):
        text = "Before.\n\n~~~\ncode here\n~~~\n\nAfter."
        result = extract_prose(text)
        assert "code here" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_indented_code(self):
        text = "Paragraph.\n\n    indented code line\n    another line\n\nNext paragraph."
        result = extract_prose(text)
        assert "indented code" not in result
        assert "Paragraph." in result
        assert "Next paragraph." in result

    def test_strips_latex_block(self):
        text = "Before math.\n\n$$E=mc^2$$\n\nAfter math."
        result = extract_prose(text)
        assert "E=mc" not in result
        assert "Before math." in result
        assert "After math." in result

    def test_strips_latex_inline(self):
        text = r"The value \(x=5\) is important."
        result = extract_prose(text)
        assert "x=5" not in result
        assert "The value" in result
        assert "is important." in result

    def test_strips_latex_bracket_block(self):
        text = "Before.\n\n\\[f(x) = x^2\\]\n\nAfter."
        result = extract_prose(text)
        assert "f(x)" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_images(self):
        text = "See the diagram: ![alt text](img.png) below."
        result = extract_prose(text)
        assert "![" not in result
        assert "img.png" not in result
        assert "See the diagram:" in result
        assert "below." in result

    def test_converts_links_to_text(self):
        text = "Visit [my site](https://example.com) for more."
        result = extract_prose(text)
        assert result == "Visit my site for more."

    def test_strips_hugo_shortcodes(self):
        text = 'Before. {{< tts src="audio.opus" >}} After.'
        result = extract_prose(text)
        assert "{{<" not in result
        assert "tts" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_self_closing_shortcodes(self):
        text = 'Check {{< relurl "path" />}} here.'
        result = extract_prose(text)
        assert "{{<" not in result
        assert "relurl" not in result
        assert "Check" in result
        assert "here." in result

    def test_strips_shortcodes_with_body(self):
        text = "Before.\n\n{{< details >}}\nHidden content\n{{< /details >}}\n\nAfter."
        result = extract_prose(text)
        assert "details" not in result
        assert "Hidden content" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_html_tags(self):
        text = "<strong>bold</strong> and <em>italic</em>"
        result = extract_prose(text)
        assert "<strong>" not in result
        assert "</strong>" not in result
        assert "bold" in result
        assert "italic" in result

    def test_strips_bold_italic_markers(self):
        text = "This is **bold** and *italic* text."
        result = extract_prose(text)
        assert "**" not in result
        assert result == "This is bold and italic text."

    def test_strips_inline_code(self):
        text = "Use the `print()` function."
        result = extract_prose(text)
        assert "`" not in result
        assert "print()" in result

    def test_collapses_whitespace(self):
        text = "First.\n\n\n\n\nSecond."
        result = extract_prose(text)
        # 3+ newlines should become at most 2
        assert "\n\n\n" not in result
        assert "First.\n\nSecond." == result

    def test_strips_horizontal_rules(self):
        text = "Above.\n\n---\n\nBelow."
        result = extract_prose(text)
        # The --- should be removed
        assert "---" not in result
        assert "Above." in result
        assert "Below." in result

    def test_strips_horizontal_rules_asterisk(self):
        text = "Above.\n\n***\n\nBelow."
        result = extract_prose(text)
        assert "***" not in result

    def test_strips_horizontal_rules_underscore(self):
        text = "Above.\n\n___\n\nBelow."
        result = extract_prose(text)
        assert "___" not in result

    def test_strips_blockquotes(self):
        text = "> This is a quote."
        result = extract_prose(text)
        assert result.strip() == "This is a quote."

    def test_strips_list_markers_dash(self):
        text = "- item one\n- item two"
        result = extract_prose(text)
        assert "- " not in result
        assert "item one" in result
        assert "item two" in result

    def test_strips_list_markers_asterisk(self):
        text = "* item one\n* item two"
        result = extract_prose(text)
        assert "* " not in result
        assert "item one" in result

    def test_strips_list_markers_ordered(self):
        text = "1. first\n2. second\n10. tenth"
        result = extract_prose(text)
        assert "1. " not in result
        assert "10. " not in result
        assert "first" in result
        assert "tenth" in result

    def test_realistic_post(self):
        """A full markdown document with headings, code, math, images, links, shortcodes."""
        md = """\
## Introduction

This post explains **gradient descent** in machine learning.

Consider the loss function:

$$L(\\theta) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - f(x_i; \\theta))^2$$

The update rule is \\(\\theta \\leftarrow \\theta - \\alpha \\nabla L\\).

```python
def gradient_descent(f, x, lr=0.01):
    for _ in range(100):
        x -= lr * grad(f, x)
    return x
```

Here is a diagram:

![gradient descent visualization](images/gd.png)

For more details, read [this paper](https://arxiv.org/abs/1234.5678) by the authors.

{{< tts src="intro.opus" >}}

> Machine learning is the future.

---

### Steps

1. Initialize parameters
2. Compute gradient
3. Update parameters

- Repeat until convergence
- Check validation loss

Use the `Adam` optimizer for better results.

<div class="note">This is an HTML note.</div>

{{< details >}}
Hidden implementation details here.
{{< /details >}}

    # This is indented code
    should_be_removed = True

That concludes the tutorial.
"""
        result = extract_prose(md)

        # Readable content should be present
        assert "Introduction" in result
        assert "gradient descent" in result
        assert "machine learning" in result
        assert "Initialize parameters" in result
        assert "Repeat until convergence" in result
        assert "Adam" in result
        assert "That concludes the tutorial." in result
        assert "this paper" in result
        assert "Machine learning is the future." in result

        # Non-readable content should be stripped
        assert "```" not in result
        assert "def gradient_descent" not in result
        assert "$$" not in result
        assert "L(\\theta)" not in result
        assert "\\(" not in result
        assert "![" not in result
        assert "gd.png" not in result
        assert "https://arxiv.org" not in result
        assert "{{<" not in result
        assert "tts src" not in result
        assert "<div" not in result
        assert "##" not in result
        assert "---" not in result
        assert "> " not in result
        assert "1. " not in result
        assert "- " not in result
        assert "**" not in result
        assert "`Adam`" not in result  # backticks stripped, text preserved
        assert "Hidden implementation details" not in result
        assert "should_be_removed" not in result
