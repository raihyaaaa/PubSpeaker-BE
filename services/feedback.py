"""Feedback generation service with phoneme-level pronunciation analysis."""

from typing import List, Dict, Optional, FrozenSet

from config import PACE_VERY_SLOW, PACE_GOOD_LOWER, PACE_MODERATE_LOWER
from utils.phonetics import arpabet_to_ipa, arpabet_to_readable
from utils.text import is_junk_token

# Same set used in app.py — function words to skip in pronunciation feedback.
_FUNCTION_WORDS: FrozenSet[str] = frozenset({
    # Determiners / articles
    "the", "a", "an",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "by", "from", "with",
    "into", "over", "under", "through", "between", "before",
    "out", "off", "down", "away", "back", "here", "there",
    "around", "across", "along", "among", "against", "during",
    "without", "within", "upon", "toward", "towards", "beyond",
    "aside", "after", "until", "since", "above", "below",
    # Copulas / auxiliaries
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "has", "have", "had",
    # Pronouns
    "it", "its",
    "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their",
    "this", "that", "these", "those",
    "i",
    # Negation / quantifiers / determiners
    "not", "no",
    "some", "any", "all", "each", "every", "much", "many",
    "more", "most", "other", "another", "such",
    # Adverbs commonly reduced in connected speech
    "if", "then", "as", "up",
    "because", "very", "like", "just", "also", "even", "still",
    "about", "only", "really", "sometimes", "already",
    "maybe", "actually", "usually", "when", "will",
    "what", "where", "how", "who", "which", "than", "too",
    # Modals
    "can", "could", "would", "should", "may", "might",
    # Common verbs naturally reduced
    "get", "got", "going", "go", "come", "came",
    # Common content words with highly variable natural pronunciation
    "although", "however", "overall", "offer", "color", "colour",
    "lives", "while", "new", "one", "now", "well", "know",
    "used", "being", "want", "own", "world", "year", "make",
    "people", "right", "think", "good", "said", "its",
})


class FeedbackService:
    """Service for generating user-friendly feedback on speech analysis."""
    
    def generate(
        self,
        transcript: str,
        pronunciation_analysis: Dict,  # New: from PronunciationAlignmentService
        grammar_issues: List[str],
        words: List[Dict]
    ) -> Dict[str, any]:
        """
        Generate dynamic feedback based on analysis results.
        
        Args:
            transcript: Original transcript text
            pronunciation_analysis: Phoneme-level alignment results
            grammar_issues: List of words with grammar issues
            words: List of word timing information
            
        Returns:
            Dictionary with feedback lists and metadata
        """
        feedback = {
            "pronunciation": self._generate_pronunciation_feedback(pronunciation_analysis),
            "grammar": self._generate_grammar_feedback(grammar_issues),
            "pace": self._generate_pace_feedback(words),
            "pronunciation_metadata": {
                "analysis_available": pronunciation_analysis.get("available", False),
                "disclaimer": pronunciation_analysis.get("disclaimer", ""),
                "confidence_scores": pronunciation_analysis.get("confidence_scores", {})
            }
        }
        
        return feedback
    
    def _generate_pronunciation_feedback(self, pronunciation_analysis: Dict) -> List[Dict]:
        """
        Generate pronunciation feedback from phoneme-level analysis.
        
        NEW: Uses forced alignment deviations, not dictionary lookup.
        """
        feedback = []
        
        if not pronunciation_analysis.get("available"):
            # MFA not available - inform user
            feedback.append({
                "type": "info",
                "message": "Pronunciation analysis requires Montreal Forced Aligner (MFA)."
            })
            if "installation_guide" in pronunciation_analysis:
                feedback.append({
                    "type": "info",
                    "message": pronunciation_analysis["installation_guide"]
                })
            return feedback
        
        all_deviations = pronunciation_analysis.get("deviations", [])
        # Filter out function words — they are naturally reduced in speech.
        # Severity-based filtering is no longer applied here; the alignment
        # service already gates on the threshold + duration penalty.
        deviations = [
            d for d in all_deviations
            if d.get("word", "").strip().lower() not in _FUNCTION_WORDS
            and not is_junk_token(d.get("word", ""))
        ]
        
        if not deviations:
            feedback.append({
                "type": "positive",
                "message": "Excellent! Your pronunciation shows strong clarity."
            })
            return feedback
        
        # Group by severity
        notable = [d for d in deviations if d["severity"] == "notable"]
        moderate = [d for d in deviations if d["severity"] == "moderate"]
        minor = [d for d in deviations if d["severity"] == "minor"]
        
        # Summary message
        if notable:
            feedback.append({
                "type": "warning",
                "message": f"Found {len(notable)} words with notable pronunciation deviations."
            })
        elif moderate:
            feedback.append({
                "type": "info",
                "message": f"Found {len(moderate)} words with moderate pronunciation deviations."
            })
        else:
            feedback.append({
                "type": "positive",
                "message": "Your pronunciation is clear with only minor variations."
            })
        
        # Specific word feedback (top 3 notable)
        for dev in notable[:3]:
            expected_ipa = arpabet_to_ipa(dev.get('expected'))
            actual_ipa = arpabet_to_ipa(dev.get('actual'))
            expected_readable = dev.get('expected_readable') or arpabet_to_readable(dev.get('expected')) or ''
            actual_readable = dev.get('actual_readable') or arpabet_to_readable(dev.get('actual')) or ''
            expected_disp = expected_ipa or dev.get('expected') or ''
            actual_disp = actual_ipa or dev.get('actual') or ''
            feedback.append({
                "type": "warning",
                "message": (
                    f"'{dev['word']}' - Expected: {expected_readable} {expected_disp}, "
                    f"Detected: {actual_readable} {actual_disp} "
                    f"(similarity: {int(dev['similarity']*100)}%)"
                )
            })
        
        return feedback
    
    def _generate_grammar_feedback(self, grammar_issues: List[str]) -> List[Dict]:
        """Generate grammar feedback messages."""
        feedback = []
        
        if not grammar_issues:
            feedback.append({
                "type": "positive",
                "message": "Your grammar is excellent!"
            })
        elif len(grammar_issues) <= 2:
            feedback.append({
                "type": "info",
                "message": "Your grammar is mostly correct."
            })
            feedback.append({
                "type": "warning",
                "message": f"Check: {', '.join(set(grammar_issues))} - see Suggestions tab for corrections."
            })
        else:
            feedback.append({
                "type": "warning",
                "message": f"Found {len(set(grammar_issues))} grammar issues."
            })
            feedback.append({
                "type": "warning",
                "message": f"Review these words: {', '.join(list(set(grammar_issues))[:5])}. See Suggestions tab."
            })
        
        return feedback
    
    def _generate_pace_feedback(self, words: List[Dict]) -> List[Dict]:
        """
        Generate speaking pace feedback based on words per minute.
        
        FIXED: Now excludes pauses >1.5s from active speaking time.
        Previously: Used total duration (first_word.start to last_word.end)
        Problem: 20 words with 10s pause = 48 WPM (should be ~120 WPM)
        """
        feedback = []
        
        if not words or len(words) == 0:
            feedback.append({
                "type": "info",
                "message": "Speak longer passages to get pace feedback."
            })
            return feedback
        
        # Find first and last words with valid timestamps
        first_word = next((w for w in words if w.get("start") is not None), None)
        last_word = next((w for w in reversed(words) if w.get("end") is not None), None)
        
        if not first_word or not last_word:
            feedback.append({
                "type": "info",
                "message": "Keep practicing to develop a natural speaking rhythm."
            })
            return feedback
        
        # Calculate active speaking duration (exclude pauses >1.5s)
        PAUSE_THRESHOLD = 1.5  # seconds
        active_duration = 0.0
        
        for i, word in enumerate(words):
            if not word.get("start") or not word.get("end"):
                continue
            
            # Add word duration
            word_duration = word["end"] - word["start"]
            active_duration += word_duration
            
            # Check gap to next word
            if i < len(words) - 1:
                next_word = words[i + 1]
                if next_word.get("start"):
                    gap = next_word["start"] - word["end"]
                    # Include gap if < threshold (normal speech pause)
                    if gap < PAUSE_THRESHOLD:
                        active_duration += gap
                    # Else: exclude long pause from calculation
        
        if active_duration <= 0:
            feedback.append({
                "type": "info",
                "message": "Keep practicing to develop a natural speaking rhythm."
            })
            return feedback
        
        # Calculate words per minute from active speaking time only
        words_per_minute = (len(words) / active_duration) * 60
        wpm_int = int(words_per_minute)
        
        if words_per_minute < PACE_VERY_SLOW:
            feedback.append({
                "type": "info",
                "message": f"Your pace is quite slow ({wpm_int} words/min). Try speaking a bit faster."
            })
        elif words_per_minute < PACE_GOOD_LOWER:
            feedback.append({
                "type": "positive",
                "message": f"Your pace is good ({wpm_int} words/min) - easy to follow."
            })
        elif words_per_minute < PACE_MODERATE_LOWER:
            feedback.append({
                "type": "info",
                "message": f"Your pace is moderate ({wpm_int} words/min). Consider pausing for emphasis."
            })
        else:
            feedback.append({
                "type": "warning",
                "message": f"Your pace is fast ({wpm_int} words/min). Try slowing down and pausing more."
            })
        
        return feedback
