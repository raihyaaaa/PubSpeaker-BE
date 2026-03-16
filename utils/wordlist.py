"""Wordlist management and word validation utilities."""

import os
import pronouncing
import difflib
import editdistance
from typing import Tuple, Optional, Set

from config import COMMON_INFLECTIONS, SIMILARITY_CUTOFF, MAX_EDIT_DISTANCE, MIN_WORD_LENGTH_FOR_MISPRONUNCIATION, MIN_WORD_LENGTH_FOR_CONCATENATION


_WORDLIST_CACHE: Optional[Set[str]] = None


# Tokens that frequently appear in speech transcripts but are not reliably
# represented in dictionary wordlists (fillers, colloquialisms, etc.).
_ALLOWED_NONSTANDARD = {
    "um",
    "uh",
    "erm",
    "yeah",
    "yep",
    "nope",
    "okay",
    "ok",
    "gonna",
    "wanna",
    "kinda",
    "sorta",
    "lemme",
    "gotta",
}


def get_wordlist() -> Set[str]:
    """
    Get the wordlist (lazy-loaded singleton).
    Tries multiple sources: CMU dict, system dictionary, fallback list.
    
    Returns:
        Set of valid words in lowercase
    """
    global _WORDLIST_CACHE
    
    if _WORDLIST_CACHE is not None:
        return _WORDLIST_CACHE
    
    # Try CMU Pronouncing Dictionary first
    try:
        cmu_dict = getattr(pronouncing, "dict")()
        if cmu_dict and isinstance(cmu_dict, dict):
            _WORDLIST_CACHE = set(cmu_dict.keys())
            return _WORDLIST_CACHE
    except Exception:
        pass
    
    # Try system dictionary
    try:
        system_dict_path = "/usr/share/dict/words"
        if os.path.exists(system_dict_path):
            with open(system_dict_path, "r", encoding="utf-8", errors="ignore") as f:
                _WORDLIST_CACHE = set(w.strip().lower() for w in f if w.strip())
                return _WORDLIST_CACHE
    except Exception:
        pass
    
    # Fallback to minimal wordlist
    _WORDLIST_CACHE = set([
        "tomorrow", "lazy", "dog", "nuts", "park", "again", "squirrel", 
        "fox", "backyard", "animals", "jump", "jumps", "will", "they"
    ])
    return _WORDLIST_CACHE


def get_base_form(word: str) -> Tuple[Optional[str], bool]:
    """
    Try to get base form of a word by removing common inflections.
    
    Args:
        word: The word to analyze
        
    Returns:
        Tuple of (base_form, is_inflected). Returns (None, False) if not found.
    """
    wordlist = get_wordlist()
    word_lower = word.lower()
    
    if word_lower in wordlist:
        return word_lower, False
    
    # Try removing common suffixes
    for suffix in ['ing', 'ed', 'es', 's', 'er', 'est', 'ly', 'd']:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            base = word_lower[:-len(suffix)]
            if base in wordlist:
                return base, True
            # For 'ed' endings, try adding 'e' back (e.g., 'liked' -> 'like')
            if suffix == 'ed':
                base_with_e = base + 'e'
                if base_with_e in wordlist:
                    return base_with_e, True
    
    return None, False


def check_mispronunciation(word: str, word_normalized: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Determine if a word is likely mispronounced.
    
    Args:
        word: Original word (with punctuation)
        word_normalized: Normalized word (lowercase, no punctuation)
        
    Returns:
        Tuple of (is_mispronounced, suggested_word, reason)
    """
    # Heuristic guardrails to reduce false positives on transcript text.
    # Contractions are often normalized into out-of-vocabulary tokens (e.g., I'm -> im).
    if "'" in word or "’" in word:
        return False, None, None

    # Likely proper noun / named entity (capital letters), skip transcript-based flagging.
    if any(c.isupper() for c in word):
        return False, None, None

    # Common spoken fillers and colloquialisms.
    if word_normalized in _ALLOWED_NONSTANDARD:
        return False, None, None

    wordlist = get_wordlist()
    
    # Check if it's a valid word
    if word_normalized in wordlist:
        return False, None, None
    
    # Check if it's a valid inflection
    base, is_inflected = get_base_form(word_normalized)
    if base:
        return False, None, None
    
    # Check for close matches with high similarity threshold
    close_matches = difflib.get_close_matches(
        word_normalized, 
        wordlist, 
        n=1, 
        cutoff=SIMILARITY_CUTOFF
    )
    
    if close_matches:
        distance = editdistance.eval(word_normalized, close_matches[0])
        if distance <= MAX_EDIT_DISTANCE and len(word_normalized) >= MIN_WORD_LENGTH_FOR_MISPRONUNCIATION:
            return True, close_matches[0], f"likely mispronunciation of '{close_matches[0]}'"
    
    # Check for word concatenations (only for longer words)
    if len(word_normalized) >= MIN_WORD_LENGTH_FOR_CONCATENATION:
        for i in range(3, len(word_normalized) - 2):
            left = word_normalized[:i]
            right = word_normalized[i:]
            
            if right in COMMON_INFLECTIONS:
                continue
                
            if left in wordlist and right in wordlist:
                return True, f"{left} {right}", f"appears to be '{left} {right}' run together"
    
    # Not in dictionary but no clear correction
    return True, None, "word not in dictionary (possible proper noun or specialized term)"
