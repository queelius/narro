"""LLM-powered paragraph rewriting for TTS."""
import logging
import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Rewrite the following paragraph as natural spoken prose for "
    "text-to-speech narration. Preserve the meaning. Expand abbreviations. "
    "Spell out symbols. Remove anything that doesn't make sense when read "
    "aloud. Keep it about the same length. Do not add commentary or "
    "meta-text. Return only the rewritten paragraph."
)

def rewrite_paragraphs(paragraphs, api_url, api_key=None, model=None):
    """Rewrite paragraphs to conversational speech prose via LLM.
    Each paragraph rewritten independently to preserve 1:1 index mapping.
    Returns list of same length. Falls back to original on error."""
    if not paragraphs:
        return []
    url = f"{api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    results = []
    for paragraph in paragraphs:
        try:
            body = {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": paragraph},
            ]}
            if model:
                body["model"] = model
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            rewritten = data["choices"][0]["message"]["content"].strip()
            results.append(rewritten)
        except Exception as e:
            logger.warning("LLM rewrite failed, using original: %s", e)
            results.append(paragraph)
    return results
