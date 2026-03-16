"""Text-to-speech service using Google TTS."""

import os
import re
from gtts import gTTS
from typing import Dict, Optional, List
from PyMultiDictionary import MultiDictionary

from config import TTS_DIR
from services.pronunciation import PronunciationService
from utils.phonetics import arpabet_to_readable


# ARPABET to IPA mapping (US English)
ARPABET_TO_IPA = {
    # Vowels
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ',
    'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɜr', 'EY': 'eɪ', 'IH': 'ɪ',
    'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
    # Consonants
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f',
    'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l',
    'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'r',
    'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}


class TTSService:
    """Service for text-to-speech generation and pronunciation examples."""
    
    def __init__(self, pronunciation_service: PronunciationService):
        """
        Initialize TTS service.
        
        Args:
            pronunciation_service: Service for getting canonical phonemes
        """
        self.pronunciation_service = pronunciation_service
        self.dictionary = MultiDictionary()
        os.makedirs(TTS_DIR, exist_ok=True)
    
    def get_or_create_audio(self, word: str) -> Dict[str, any]:
        """
        Get or create TTS audio for a word.
        
        If audio file doesn't exist, generates it using Google TTS.
        
        Args:
            word: The word to generate audio for
            
        Returns:
            Dictionary with 'word', 'listen_url', and 'canonical' phonemes
        """
        safe_filename = self._sanitize_filename(word)
        audio_filename = f"{safe_filename}.mp3"
        audio_path = os.path.join(TTS_DIR, audio_filename)
        
        # Generate audio if it doesn't exist
        if not os.path.exists(audio_path):
            tts = gTTS(text=word, lang="en", slow=False)
            tts.save(audio_path)
        
        canonical_phonemes = self.pronunciation_service.get_canonical_phonemes(word)
        
        # Convert ARPABET to IPA format
        ipa_pronunciation = self._convert_to_ipa(canonical_phonemes) if canonical_phonemes else None
        
        # Convert ARPABET to human-readable guide (e.g. "AW·fuhn")
        readable_pronunciation = arpabet_to_readable(canonical_phonemes) if canonical_phonemes else None
        
        # Get word definition
        definition = self._get_definition(word)
        
        return {
            "word": word,
            "listen_url": f"/tts/{audio_filename}",
            "canonical": ipa_pronunciation,
            "readable": readable_pronunciation,
            "definition": definition
        }
    
    def _sanitize_filename(self, word: str) -> str:
        """
        Sanitize a word for use as a filename.
        
        Args:
            word: The word to sanitize
            
        Returns:
            Safe filename string
        """
        return "".join(c for c in word if c.isalnum() or c in "-_")
    
    def _convert_to_ipa(self, phonemes: List[str]) -> str:
        """
        Convert ARPABET phonemes to IPA notation.
        
        Args:
            phonemes: List of ARPABET phonemes (e.g., ['B', 'AH0', 'N', 'AE1', 'N', 'AH0'])
            
        Returns:
            IPA formatted string (e.g., '/bəˈnæn.ə/')
        """
        if not phonemes:
            return None
        
        ipa_chars = []
        for phoneme in phonemes:
            # Remove stress markers (0, 1, 2)
            base_phoneme = re.sub(r'[012]', '', phoneme)
            stress = re.search(r'[012]', phoneme)
            
            # Convert to IPA
            if base_phoneme in ARPABET_TO_IPA:
                ipa_char = ARPABET_TO_IPA[base_phoneme]
                
                # Add primary stress marker before vowel
                if stress and stress.group() == '1':
                    ipa_chars.append('ˈ')
                # Add secondary stress marker before vowel
                elif stress and stress.group() == '2':
                    ipa_chars.append('ˌ')
                
                ipa_chars.append(ipa_char)
        
        # Join and format with slashes
        ipa_string = ''.join(ipa_chars)
        
        # Google TTS uses US English by default
        return f"/{ipa_string}/"
    
    def _get_definition(self, word: str) -> str:
        """
        Get the definition of a word.
        
        PyMultiDictionary only has entries for **base forms** — so
        ``smartphones`` won't be found but ``smartphone`` will.  We try
        the original word first, then progressively strip common English
        inflectional suffixes until we get a hit.
        
        Args:
            word: The word to define
            
        Returns:
            Definition string, or a message if not found
        """
        candidates = self._lemma_candidates(word)
        for attempt in candidates:
            result = self._lookup_definition(attempt)
            if result:
                return result
        return "Definition not found"

    def _lemma_candidates(self, word: str) -> List[str]:
        """Return a list of candidate base-forms to look up, ordered from
        most-specific to most-general.

        Examples::

            smartphones → [smartphones, smartphone, smartphones (lower), smartphone (lower)]
            healthcare  → [healthcare, health care, healthcare (lower), ...]
            connected   → [connected, connect, ...]
            devices     → [devices, device, ...]
            algorithms  → [algorithms, algorithm, ...]
            lives       → [lives, live, life, ...]
        """
        seen = []

        def _add(w: str):
            if w and w not in seen:
                seen.append(w)

        _add(word)
        lo = word.lower()
        _add(lo)

        # ── Inflectional suffixes (English) ──────────────────────
        # Order matters — try longest suffix first.
        for form in self._strip_suffixes(lo):
            _add(form)

        # ── Compound words: try splitting on common join points ──
        # "healthcare" → "health care", "smartphones" → "smart phones"
        # Only for words >= 8 chars to avoid false splits.
        if len(lo) >= 8 and "-" not in lo:
            for i in range(3, len(lo) - 2):
                left, right = lo[:i], lo[i:]
                if len(left) >= 3 and len(right) >= 3:
                    _add(f"{left} {right}")

        return seen

    @staticmethod
    def _strip_suffixes(word: str) -> List[str]:
        """Return candidate base-forms by stripping English inflections."""
        forms: List[str] = []
        w = word

        # -ies → -y  (e.g. "technologies" → "technology")
        if w.endswith("ies") and len(w) > 4:
            forms.append(w[:-3] + "y")

        # -ves → -fe / -f  (e.g. "lives" → "life", "knives" → "knife")
        if w.endswith("ves") and len(w) > 4:
            forms.append(w[:-3] + "fe")
            forms.append(w[:-3] + "f")

        # -ses / -xes / -zes / -ches / -shes → drop -es
        if w.endswith("es") and len(w) > 3:
            if w.endswith(("ses", "xes", "zes", "ches", "shes")):
                forms.append(w[:-2])
            else:
                forms.append(w[:-1])   # -es → -e  (e.g. "devices" → "device")
                forms.append(w[:-2])   # -es →     (e.g. "diagnoses" → "diagnos…")

        # -s (simple plural, e.g. "smartphones" → "smartphone")
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            forms.append(w[:-1])

        # -ed  (e.g. "connected" → "connect")
        if w.endswith("ed") and len(w) > 4:
            forms.append(w[:-2])       # connected → connect
            forms.append(w[:-1])       # e.g. "used" → "use"  (drop d only is wrong, skip)
            if w[-3] == w[-4]:         # "stopped" → "stop" (double consonant)
                forms.append(w[:-3])

        # -ing  (e.g. "running" → "run")
        if w.endswith("ing") and len(w) > 5:
            forms.append(w[:-3])
            forms.append(w[:-3] + "e")  # "driving" → "drive"
            if len(w) > 6 and w[-4] == w[-5]:  # "running" → "run"
                forms.append(w[:-4])

        # -ly  (e.g. "potentially" → "potential")
        if w.endswith("ly") and len(w) > 4:
            forms.append(w[:-2])
            if w.endswith("ily"):
                forms.append(w[:-3] + "y")  # "happily" → "happy"

        # -er  (e.g. "faster" → "fast")
        if w.endswith("er") and len(w) > 4:
            forms.append(w[:-2])
            forms.append(w[:-1])  # "nicer" → "nice"

        # -est  (e.g. "fastest" → "fast")
        if w.endswith("est") and len(w) > 5:
            forms.append(w[:-3])

        return forms

    def _lookup_definition(self, word: str) -> Optional[str]:
        """Try to look up *word* in PyMultiDictionary.  Returns a formatted
        definition string on success, or ``None`` on failure."""
        try:
            meaning = self.dictionary.meaning('en', word)
            if not meaning:
                return None

            # --- dict format: {"Noun": [...], "Verb": [...]} ---
            if isinstance(meaning, dict):
                parts = []
                for pos, defs in meaning.items():
                    if isinstance(defs, list) and defs:
                        parts.append(f"({pos}) {defs[0]}")
                    elif isinstance(defs, str) and defs:
                        parts.append(f"({pos}) {defs}")
                if parts:
                    return " | ".join(parts)

            # --- tuple format: ([POS_list], def1, def2, ...) ---
            if isinstance(meaning, tuple) and len(meaning) >= 2:
                parts_of_speech = meaning[0] if meaning[0] else []
                if parts_of_speech:
                    definitions = []
                    for i, pos in enumerate(parts_of_speech):
                        if len(meaning) > i + 1 and meaning[i + 1]:
                            def_text = meaning[i + 1]
                            if isinstance(def_text, list):
                                def_text = def_text[0] if def_text else ""
                            if def_text:
                                definitions.append(f"({pos}) {def_text}")
                    if definitions:
                        return " | ".join(definitions)

            # --- list format: [def_string, ...] ---
            if isinstance(meaning, list) and meaning:
                first = meaning[0]
                if isinstance(first, str) and first:
                    return first

        except Exception as e:
            print(f"Error getting definition for '{word}': {e}")
        return None
    
    def get_audio_path(self, filename: str) -> Optional[str]:
        """
        Get the full path to a TTS audio file.
        
        Args:
            filename: The audio filename
            
        Returns:
            Absolute path to audio file, or None if doesn't exist
        """
        path = os.path.join(TTS_DIR, filename)
        return path if os.path.exists(path) else None
