"""Utility functions for text processing and annotation."""

import re
import string
import difflib
from typing import Dict, List, Optional, Set, Tuple, Union


def normalize_token(token: str) -> str:
    """
    Normalize a token by removing punctuation and converting to lowercase.
    
    Args:
        token: The token to normalize
        
    Returns:
        Normalized token string
    """
    return token.lower().strip(string.punctuation + "\"\"\"''")


# Tokens emitted by Whisper / ASR that are not real words.
_ASR_JUNK_TOKENS: Set[str] = {"<unk>", "[unk]", "<blank>", "[blank]"}


def is_junk_token(token: str) -> bool:
    """Return True if *token* is a known ASR artefact (e.g. ``<unk>``)."""
    return token.strip().lower() in _ASR_JUNK_TOKENS


def _split_sentences(text: str) -> List[dict]:
    """Split *text* into sentences preserving original character offsets.

    Returns a list of dicts: ``{"text": str, "start": int, "end": int}``.
    Sentence boundaries are ``.``, ``!``, ``?`` (followed by whitespace or
    end-of-string).  If there are no such boundaries the whole text is
    returned as a single sentence.
    """
    sentences = []
    for m in re.finditer(r'[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$', text):
        sentences.append({
            "text": m.group(),
            "start": m.start(),
            "end": m.end(),
        })
    if not sentences:
        sentences.append({"text": text, "start": 0, "end": len(text)})
    return sentences


def _resolve_mispronounced_indices(
    orig_tokens: List[str],
    mispronounced_words: List[Union[str, Dict]],
    words_with_timestamps: Optional[List[Dict]] = None,
) -> Set[int]:
    """Return the set of **token indices** (into *orig_tokens*) that should
    be highlighted as mispronounced.

    *mispronounced_words* may be:
      - A plain ``List[str]`` of word strings — legacy format.  When
        ``words_with_timestamps`` is available we match each mispronounced
        word to the nearest token by timestamp so only ONE occurrence is
        highlighted, not all of them.
      - A ``List[dict]`` with ``{"word": str, "start": float, "end": float}``
        which carries the position already.
    """
    if not mispronounced_words:
        return set()

    orig_norm = [normalize_token(t) for t in orig_tokens]

    # ── Fast path: dict-style items with timestamps ──────────────
    sample = mispronounced_words[0]
    if isinstance(sample, dict):
        indices: Set[int] = set()
        for item in mispronounced_words:
            w_norm = normalize_token(item.get("word", ""))
            start = item.get("start")
            end = item.get("end")
            if not w_norm:
                continue
            best_idx = _find_best_token_index(orig_norm, w_norm, start, end, words_with_timestamps)
            if best_idx is not None:
                indices.add(best_idx)
        return indices

    # ── Legacy: plain list of word strings ───────────────────────
    if words_with_timestamps:
        # Build a timestamp-indexed lookup for each transcript token
        indices = set()
        for word_str in mispronounced_words:
            w_norm = normalize_token(word_str)
            if not w_norm or is_junk_token(word_str):
                continue
            best_idx = _find_best_token_index(orig_norm, w_norm, None, None, words_with_timestamps)
            if best_idx is not None:
                indices.add(best_idx)
        return indices

    # Absolute fallback (no timestamps): highlight FIRST occurrence only
    indices = set()
    seen: Set[str] = set()
    for idx, tok_norm in enumerate(orig_norm):
        for word_str in mispronounced_words:
            w_norm = normalize_token(word_str)
            if w_norm and tok_norm == w_norm and w_norm not in seen:
                indices.add(idx)
                seen.add(w_norm)
                break  # only first occurrence
    return indices


def _find_best_token_index(
    orig_norm: List[str],
    word_norm: str,
    start: Optional[float],
    end: Optional[float],
    words_with_timestamps: Optional[List[Dict]],
) -> Optional[int]:
    """Find the best matching token index for a word, using timestamps if
    available to disambiguate multiple occurrences."""
    candidates = [i for i, t in enumerate(orig_norm) if t == word_norm]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Multiple occurrences — use timestamps to pick the right one
    if start is not None and words_with_timestamps:
        best = None
        best_dist = float("inf")
        for idx in candidates:
            if idx < len(words_with_timestamps):
                ts = words_with_timestamps[idx]
                ts_start = ts.get("start")
                if ts_start is not None:
                    dist = abs(ts_start - start)
                    if dist < best_dist:
                        best_dist = dist
                        best = idx
        if best is not None:
            return best

    # Fallback: just the first occurrence
    return candidates[0]


def annotate_transcript(
    original: str,
    corrected: str,
    mispronounced_words: List[Union[str, Dict]],
    grammar_issues: List[str] = None,
    words_with_timestamps: Optional[List[Dict]] = None,
) -> Tuple[str, List[str]]:
    """
    Annotate transcript with markup for mispronunciations and grammar errors.

    Uses an **enhanced word-level diff** between the original and corrected
    text as the single source of truth for grammar highlighting.  All three
    edit operations are covered:

    * **Substitution** (``replace``)  → the changed word is highlighted.
    * **Deletion** (``delete``)       → the removed word is highlighted.
    * **Insertion** (``insert``)      → the words flanking the gap are
      highlighted, because the original phrase is wrong *due to a missing
      word* (e.g. "like eat" → "like **to** eat").

    ``autojunk=False`` is used so that frequent function words like "the",
    "a", "to" are never silently skipped by the heuristic.

    Returns:
        ``(annotated_string, grammar_issue_words)``
    """
    # ── 1. Identify which tokens are mispronounced (by index) ───────
    orig_tokens = original.split()
    mispron_indices = _resolve_mispronounced_indices(
        orig_tokens, mispronounced_words, words_with_timestamps
    )

    # ── 2. Diff original vs corrected to find grammar changes ───────
    corr_tokens = corrected.split()
    orig_norm = [normalize_token(t) for t in orig_tokens]
    corr_norm = [normalize_token(t) for t in corr_tokens]

    grammar_token_indices: set = set()

    # autojunk=False prevents SequenceMatcher from ignoring high-frequency
    # tokens (articles, prepositions, copulas) — exactly the words ESL
    # speakers most often omit.
    matcher = difflib.SequenceMatcher(
        a=orig_norm, b=corr_norm, autojunk=False,
    )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag in ("replace", "delete"):
            orig_seg = orig_norm[i1:i2]
            corr_seg = corr_norm[j1:j2]

            # Skip compound-word merges ("cyber security" → "cybersecurity")
            if (len(orig_seg) >= 2 and len(corr_seg) == 1
                    and "".join(orig_seg) == corr_seg[0]):
                continue

            for idx in range(i1, i2):
                if orig_tokens[idx].count("-") >= 2:
                    continue                     # ASR artifact
                grammar_token_indices.add(idx)

        elif tag == "insert":
            # The corrected text has word(s) that the original is MISSING.
            # Example: "like eat" → "like TO eat"  (missing infinitive)
            #          "I happy"  → "I AM happy"   (missing copula)
            #
            # Strategy: highlight the original words on both sides of the
            # gap so the user sees the problematic phrase.  After the
            # adjacent-tag merge at the end, "like" + "eat" become one
            # continuous <yellow>like eat</yellow> span.
            #
            # Skip purely punctuation insertions (e.g. adding a comma).
            inserted = corr_norm[j1:j2]
            if not any(w for w in inserted if w.strip(string.punctuation)):
                continue

            # Word before the gap
            if i1 > 0 and orig_tokens[i1 - 1].count("-") < 2:
                grammar_token_indices.add(i1 - 1)
            # Word after the gap
            if i1 < len(orig_tokens) and orig_tokens[i1].count("-") < 2:
                grammar_token_indices.add(i1)

    # ── 3. Collect the grammar-issue word list (deduplicated) ───────
    seen_words: set = set()
    grammar_word_list: List[str] = []
    for idx in sorted(grammar_token_indices):
        w = normalize_token(orig_tokens[idx])
        if w and w not in seen_words:
            seen_words.add(w)
            grammar_word_list.append(w)

    # ── 4. Build annotated output — word-level tags ─────────────────
    parts: list = []
    for idx, tok in enumerate(orig_tokens):
        if is_junk_token(tok):
            continue
        if idx in mispron_indices:
            parts.append(f"<red>{tok}</red>")
        elif idx in grammar_token_indices:
            parts.append(f"<yellow>{tok}</yellow>")
        else:
            parts.append(tok)

    result = " ".join(parts)

    # Merge adjacent same-colour tags for cleaner output:
    #   <yellow>Me</yellow> <yellow>like</yellow> → <yellow>Me like</yellow>
    result = re.sub(r'</yellow>\s+<yellow>', ' ', result)
    result = re.sub(r'</red>\s+<red>', ' ', result)

    return result, grammar_word_list


# ---------------------------------------------------------------------------
# Corrected-paragraph annotation (green highlights on changed words)
# ---------------------------------------------------------------------------

def annotate_corrected(original: str, corrected: str) -> str:
    """Build the **corrected** paragraph with ``<green>`` tags around words
    that differ from the *original*.

    The function produces the corrected text (not the original) with markup,
    so the React Native app can show:

    * Top: the original transcript (or the ``annotated`` version with red/yellow)
    * Bottom: the corrected paragraph with ``<green>changed words</green>``

    Only genuinely changed tokens are highlighted — punctuation-only
    differences (e.g. adding a comma) are **not** wrapped in green so the
    user sees meaningful grammar fixes, not noise.
    """
    if not original or not corrected:
        return corrected or original or ""

    orig_tokens = original.split()
    corr_tokens = corrected.split()

    orig_norm = [normalize_token(t) for t in orig_tokens]
    corr_norm = [normalize_token(t) for t in corr_tokens]

    # Use SequenceMatcher to align the two token lists
    matcher = difflib.SequenceMatcher(a=orig_norm, b=corr_norm, autojunk=False)

    output_parts: list = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        segment = corr_tokens[j1:j2]
        if tag == "equal":
            output_parts.extend(segment)
        elif tag in ("replace", "insert"):
            for tok in segment:
                # Only highlight if there is a real word change, not just
                # punctuation (e.g. adding a comma after a word).
                tok_norm = normalize_token(tok)
                if tok_norm:
                    output_parts.append(f"<green>{tok}</green>")
                else:
                    output_parts.append(tok)
        # "delete" — tokens removed from original; nothing to add to corrected

    result = " ".join(output_parts)

    # Merge adjacent green tags: "<green>word1</green> <green>word2</green>"
    # → "<green>word1 word2</green>" for a nicer reading experience
    result = re.sub(
        r"</green>\s+<green>",
        " ",
        result,
    )

    return result
