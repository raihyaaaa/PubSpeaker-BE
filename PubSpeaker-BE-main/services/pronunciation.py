"""Pronunciation analysis service."""

from g2p_en import G2p
import pronouncing
from typing import Dict, List, Optional

from utils.text import normalize_token
from utils.wordlist import check_mispronunciation

# ASR junk tokens to skip
_ASR_JUNK = {"<unk>", "[unk]", "<blank>", "[blank]"}


class PronunciationService:
    """Service for pronunciation analysis and phoneme generation."""
    
    def __init__(self):
        """Initialize G2P (grapheme-to-phoneme) converter."""
        print("Loading G2P model...")
        self.g2p = G2p()
        print("G2P model loaded successfully")
    
    def get_canonical_phonemes(self, word: str) -> Optional[List[str]]:
        """
        Get canonical phoneme representation for a word.
        
        Tries CMU Pronouncing Dictionary first, falls back to G2P.
        
        Args:
            word: The word to get phonemes for
            
        Returns:
            List of phoneme strings, or None if unable to generate
        """
        word_lower = word.lower()
        
        # Try CMU Pronouncing Dictionary first (more reliable)
        try:
            pronunciations = pronouncing.phones_for_word(word_lower)
            if pronunciations:
                return pronunciations[0].split()
        except Exception:
            pass
        
        # Fall back to G2P model
        try:
            phonemes = self.g2p(word_lower)
            phonemes = [p for p in phonemes if isinstance(p, str) and p.strip()]
            if phonemes:
                return phonemes
        except Exception:
            pass
        
        return None
    
    def analyze_words(self, words: List[Dict]) -> List[Dict]:
        """
        Analyze a list of words for pronunciation issues.
        
        Args:
            words: List of word dictionaries with 'word', 'start', 'end' keys
            
        Returns:
            List of mispronunciation dictionaries with details
        """
        mispronunciations = []
        
        for word_info in words:
            word_raw = word_info["word"]

            # Skip ASR junk tokens
            if word_raw.strip().lower() in _ASR_JUNK:
                continue

            word_normalized = normalize_token(word_raw)
            
            is_mispron, suggested, reason = check_mispronunciation(word_raw, word_normalized)
            
            if is_mispron and suggested:
                # Get canonical phonemes for the suggested correct word
                intended_word = suggested.split()[0]
                canonical_phonemes = self.get_canonical_phonemes(intended_word)
                
                mispronunciations.append({
                    "word": word_raw,
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                    "canonical": canonical_phonemes,
                    "suggested_intended": suggested,
                    "reason": reason
                })
            elif is_mispron and not suggested:
                # Word not in dictionary, no suggestion available
                mispronunciations.append({
                    "word": word_raw,
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                    "canonical": self.get_canonical_phonemes(word_normalized),
                    "suggested_intended": None,
                    "reason": reason
                })
        
        return mispronunciations
