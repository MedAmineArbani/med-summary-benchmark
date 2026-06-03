"""
Prompt template for the LLM+RAG variant.

Key design choices, all of which should appear in the report:

1. Retrieved knowledge is in a CLEARLY DELIMITED block, separate from the
   article. The model must not confuse "reference knowledge" with "source
   content".

2. The system prompt is EXPLICIT about the role of each block:
   - the article is the source of truth for the summary content;
   - retrieved knowledge is background context to interpret terms, NOT new
     facts to add.

3. We keep the same section structure as `structured` (Background / Methods /
   Results / Conclusions) so RAG and non-RAG outputs are comparable.
"""

from __future__ import annotations

from src.llm.prompts import Prompt


RAG_SYSTEM = (
    "You are a medical writing assistant summarizing biomedical articles "
    "for clinicians.\n\n"
    "You receive two inputs:\n"
    "- ARTICLE: the source document. ALL summary content must come from it.\n"
    "- REFERENCE KNOWLEDGE: short factual notes on medical terms that may "
    "appear in the article. Use this ONLY to interpret terms and units. "
    "Do NOT add facts from REFERENCE KNOWLEDGE that are not already in the "
    "ARTICLE.\n\n"
    "Strict rules:\n"
    "1. Base every claim on the ARTICLE.\n"
    "2. Do not introduce numbers, drugs, dosages, or findings absent from "
    "the ARTICLE.\n"
    "3. Preserve quantitative details from the ARTICLE exactly as written.\n"
    "4. Use neutral, factual language. No speculation."
)


RAG_USER_TEMPLATE = (
    "REFERENCE KNOWLEDGE (background only, do not add facts from here):\n"
    "{retrieved}\n\n"
    "ARTICLE (source of truth):\n"
    "{article}\n\n"
    "Write a structured summary of the ARTICLE in four short paragraphs, "
    "approximately 200 words total. Use exactly these sections:\n\n"
    "Background:\nMethods:\nResults:\nConclusions:\n\n"
    "Structured summary:"
)


def make_rag_prompt(retrieved_block: str) -> Prompt:
    """
    Build a Prompt object with the retrieved knowledge baked in.

    The `user_template` of the returned prompt still contains `{article}`,
    so it stays compatible with the existing generation pipeline.
    """
    user_tpl = RAG_USER_TEMPLATE.replace("{retrieved}", retrieved_block)
    return Prompt(
        name="rag_structured",
        system=RAG_SYSTEM,
        user_template=user_tpl,
    )
