from __future__ import annotations


def clean_extracted_text(text: str) -> str:
    """Preserve the historical Step 03 output used by existing LLM JSON."""
    return "\n".join(
        stripped
        for line in text.replace("\x00", " ").splitlines()
        if (stripped := line.strip())
    )
