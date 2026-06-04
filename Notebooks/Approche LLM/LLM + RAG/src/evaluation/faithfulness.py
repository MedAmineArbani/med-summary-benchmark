"""
Faithfulness metric for medical summaries.

The classic ROUGE/BERTScore metrics compare the prediction to a gold
reference. They do NOT measure whether the prediction is faithful to the
SOURCE document — and that's exactly the failure mode of LLM-based
summarization: hallucinated drugs, invented numbers, fabricated dosages.

This module measures faithfulness directly by:

  1. Extracting "risky" entities from the prediction:
     - Numbers and dosages (e.g. "18 mg/L", "n=240", "p<0.05")
     - Drug names (from a curated common-drug list)
     - Uppercase abbreviations (3+ letters: ICU, COPD, MRI, ...)
  2. Checking, for each extracted entity, whether it appears in the
     source article.
  3. Reporting the fraction of prediction entities that are supported by
     the source.

Faithfulness = supported_entities / total_entities_in_prediction

A score of 1.0 means every risky entity in the summary is grounded in the
source. A score below 1.0 means the prediction introduces at least one
piece of information not present in the source — a hallucination signal.

This is intentionally interpretable. The full list of unsupported
entities is printed in the report for qualitative analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# A curated list of common drug names and drug class suffixes. Not exhaustive,
# but covers the bulk of what PubMed clinical abstracts mention. Easy to extend.
COMMON_DRUGS = {
    # cardiovascular
    "aspirin", "warfarin", "clopidogrel", "heparin", "atorvastatin",
    "rosuvastatin", "simvastatin", "lisinopril", "enalapril", "ramipril",
    "losartan", "valsartan", "metoprolol", "bisoprolol", "carvedilol",
    "amlodipine", "nifedipine", "furosemide", "spironolactone", "digoxin",
    # diabetes / endocrine
    "metformin", "insulin", "glipizide", "glyburide", "sitagliptin",
    "empagliflozin", "liraglutide", "levothyroxine",
    # antibiotics
    "amoxicillin", "azithromycin", "ciprofloxacin", "vancomycin", "ceftriaxone",
    "doxycycline", "penicillin", "cephalexin", "clindamycin",
    # pain / anti-inflammatory
    "ibuprofen", "acetaminophen", "paracetamol", "morphine", "oxycodone",
    "tramadol", "naproxen", "diclofenac", "prednisone", "dexamethasone",
    # psych
    "fluoxetine", "sertraline", "citalopram", "venlafaxine", "haloperidol",
    "risperidone", "olanzapine", "lithium", "diazepam", "lorazepam",
    # chemo / oncology
    "cisplatin", "carboplatin", "paclitaxel", "doxorubicin", "tamoxifen",
    "rituximab", "trastuzumab",
}


# Regex for numbers, with optional unit. Captures things like:
#   18 mg/L, 0.95, 95%, p<0.05, n=240, 1,234, 2.5x10^-3
_NUMBER_PATTERN = re.compile(
    r"""
    (?<![A-Za-z])           # not preceded by a letter (avoid 'fig1')
    [-+]?                   # optional sign
    \d{1,3}(?:[,\s]\d{3})*  # integer part, possibly with thousand separators
    (?:\.\d+)?              # optional decimal part
    (?:\s*[%xX]|             # percent or "x" multiplier
       \s*[a-zA-Z/^µ]+(?:[a-zA-Z/^0-9-]*)?  # unit: mg/L, mmHg, ng/mL, ...
    )?
    """,
    re.VERBOSE,
)


# Uppercase abbreviations, 3-6 letters. We require them at word boundaries.
_ABBREV_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{2,5})\b")


# Words we don't consider "real" abbreviations even though they're uppercase
_ABBREV_STOPWORDS = {"THE", "AND", "FOR", "WITH", "FROM", "INTO", "THIS", "THAT",
                     "WERE", "WAS", "HAVE", "HAS", "ARE", "BUT", "NOT"}


@dataclass
class FaithfulnessReport:
    score: float
    total_entities: int
    supported: int
    unsupported_entities: list[str]
    entities_by_type: dict[str, list[str]]


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace. Used for case-insensitive matching."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def _extract_numbers(text: str) -> list[str]:
    candidates = _NUMBER_PATTERN.findall(text)
    out: list[str] = []
    for c in candidates:
        c = c.strip()
        # Discard pure year-like 4-digit numbers (1990-2030) — too noisy
        if re.fullmatch(r"(19|20)\d{2}", c):
            continue
        # Discard standalone single digits ('1', '2') — usually section numbers
        if re.fullmatch(r"\d", c):
            continue
        out.append(c)
    return out


def _extract_drugs(text: str) -> list[str]:
    text_norm = _normalize(text)
    found = []
    for drug in COMMON_DRUGS:
        if re.search(rf"\b{re.escape(drug)}\b", text_norm):
            found.append(drug)
    return sorted(set(found))


def _extract_abbreviations(text: str) -> list[str]:
    matches = _ABBREV_PATTERN.findall(text)
    return sorted({m for m in matches if m not in _ABBREV_STOPWORDS})


def _is_supported(entity: str, source_norm: str) -> bool:
    """An entity is supported if it appears (normalized) in the source."""
    return _normalize(entity) in source_norm


def compute_faithfulness(prediction: str, source: str) -> FaithfulnessReport:
    """
    Compare entities in `prediction` against `source`.

    Returns a FaithfulnessReport. The `score` field is the fraction of
    risky entities in `prediction` that also appear in `source`.
    """
    source_norm = _normalize(source)

    numbers = _extract_numbers(prediction)
    drugs = _extract_drugs(prediction)
    abbrevs = _extract_abbreviations(prediction)

    entities_by_type = {
        "numbers": numbers,
        "drugs": drugs,
        "abbreviations": abbrevs,
    }

    all_entities = numbers + drugs + abbrevs
    if not all_entities:
        # No risky entities -> trivially faithful (nothing to hallucinate)
        return FaithfulnessReport(
            score=1.0,
            total_entities=0,
            supported=0,
            unsupported_entities=[],
            entities_by_type=entities_by_type,
        )

    unsupported = [e for e in all_entities if not _is_supported(e, source_norm)]
    supported = len(all_entities) - len(unsupported)

    return FaithfulnessReport(
        score=supported / len(all_entities),
        total_entities=len(all_entities),
        supported=supported,
        unsupported_entities=sorted(set(unsupported)),
        entities_by_type=entities_by_type,
    )
