"""
Phase 1 sanity check.

Verifies that:
1. The `ollama` Python client is installed.
2. The Ollama daemon is reachable.
3. The Mistral model is available locally.
4. We can get a sensible medical answer back.

Run with:
    python scripts/test_ollama.py
"""

import sys

try:
    import ollama
except ImportError:
    print("ERROR: the `ollama` package is not installed.")
    print("Run:  pip install ollama")
    sys.exit(1)


SYSTEM_PROMPT = (
    "You are a medical assistant. Answer in clear, concise English. "
    "Stick to widely accepted clinical facts and do not invent details."
)

USER_PROMPT = "List 3 common symptoms of pneumonia, one per line, no extra commentary."


def main() -> None:
    print("Calling Mistral via Ollama...\n")
    try:
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )
    except Exception as exc:
        print(f"ERROR talking to Ollama: {exc}")
        print("\nChecklist:")
        print("  - Is the Ollama daemon running? (try `ollama list` in a terminal)")
        print("  - Did you run `ollama pull mistral`?")
        sys.exit(1)

    content = response["message"]["content"].strip()
    print("--- Model reply ---")
    print(content)
    print("-------------------")
    print("\nIf the answer looks like a real list of pneumonia symptoms,")
    print("Phase 1 is complete. Move on to Phase 2 (data).")


if __name__ == "__main__":
    main()
