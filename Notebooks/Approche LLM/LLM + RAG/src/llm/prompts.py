"""
Prompt templates for medical summarization with a local LLM.

We keep prompts in code (not in a YAML or random docstring) so they are
version-controlled. Each prompt is identified by a short name that ends up
in the prediction `approach` field — that way the benchmark table can
compare prompt variants directly.

Two variants are shipped:

  - "basic"      : minimal instruction, no structure constraint.
                   Tests what the LLM does "out of the box".

  - "structured" : explicit anti-hallucination guardrails + required
                   section structure (Background / Methods / Results /
                   Conclusions) matching PubMed abstracts.

The ablation between these two variants becomes a section in the report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prompt:
    name: str
    system: str
    user_template: str  # must contain "{article}"

    def render_user(self, article: str) -> str:
        return self.user_template.format(article=article)


BASIC = Prompt(
    name="basic",
    system=(
        "You are a medical assistant. Summarize the medical document the user "
        "provides in clear, concise English."
    ),
    user_template=(
        "Summarize the following medical article in about 200 words.\n\n"
        "Article:\n{article}\n\nSummary:"
    ),
)


STRUCTURED = Prompt(
    name="structured",
    system=(
        "You are a medical writing assistant. Your task is to summarize "
        "biomedical articles for clinicians.\n\n"
        "Strict rules:\n"
        "1. Base the summary ONLY on information present in the source article. "
        "Do NOT add facts, numbers, dosages, or conclusions that are not stated "
        "in the text.\n"
        "2. If a piece of information is not in the article, do not invent it.\n"
        "3. Preserve all quantitative details from the source (sample sizes, "
        "p-values, effect sizes, dosages) exactly as written.\n"
        "4. Use neutral, factual language. No speculation."
    ),
    user_template=(
        "Summarize the following biomedical article. Use exactly these four "
        "sections, each one short paragraph:\n\n"
        "Background:\nMethods:\nResults:\nConclusions:\n\n"
        "Total length: roughly 200 words.\n\n"
        "Article:\n{article}\n\nStructured summary:"
    ),
)


PROMPTS: dict[str, Prompt] = {
    "basic": BASIC,
    "structured": STRUCTURED,
}


def get_prompt(name: str) -> Prompt:
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt '{name}'. Available: {list(PROMPTS)}")
    return PROMPTS[name]
